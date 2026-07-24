const logger = require('../../config/logger');
const EventEmitter = require('events');
const util = require("util");
const delay = util.promisify(setTimeout);
const WebSocket = require('ws');

/**
 * Drop-in replacement for TextToSpeechServiceElevenLabs backed by VoxServe.
 *
 * Public interface is identical to the ElevenLabs service:
 *   - constructor(sampleRate, agentName, userName)
 *   - setVoice(voiceId)
 *   - generate({ partialResponse }, interactionId)
 *   - stop(), end()
 *   - events: 'tts_ready' (once ready), 'audio' (raw int16 PCM Buffers)
 *
 * It keeps a single persistent WebSocket open to VoxServe's /ws endpoint and
 * synthesizes one queued utterance at a time (start -> binary PCM -> end),
 * re-chunking the PCM into fixed-size frames for WebRTC, exactly like before.
 */
class TextToSpeechServiceVoxServe extends EventEmitter {
    constructor(sampleRate, agentName, userName) {
        super();
        this.isConnecting = false;
        this.isSpeaking = false;
        this.messageQueue = [];
        this.isProcessing = false;
        this.currentInteractionId = null;
        this.audioReceived = false;
        this.audioChunks = [];
        this.agentName = agentName || '';
        this.userName = userName || '';

        // Output is fixed to 16 kHz mono int16 PCM. VoxServe serves 16k/24k;
        // we always request 16k here regardless of the constructor arg.
        this.requestedSampleRate = sampleRate;
        this.sampleRate = 16000;
        if (sampleRate && sampleRate !== this.sampleRate) {
            logger.info(`${this.agentName} ${this.userName} VoxServe: forcing 16k PCM output (requested ${sampleRate})`);
        }

        // Connection config (override via env). Point at your VoxServe host.
        const host = process.env.VOX_TTS_HOST || '178.63.124.87';
        const port = process.env.VOX_TTS_PORT || '2200';
        this.wsUrl = process.env.VOX_TTS_WS_URL || `ws://${host}:${port}/ws`;

        // Default voice. Register once via POST /voices and reference by id.
        this.voice = process.env.VOX_TTS_VOICE_ID || null; // null => model default voice

        // Optional Zonos conditioning controls (replace ElevenLabs voice_settings).
        this.speakingRate = process.env.VOX_TTS_SPEAKING_RATE ? Number(process.env.VOX_TTS_SPEAKING_RATE) : undefined;
        this.pitchStd = process.env.VOX_TTS_PITCH_STD ? Number(process.env.VOX_TTS_PITCH_STD) : undefined;

        // Bytes per emitted chunk (int16 mono). 4800 B = 2400 samples
        // (150 ms @16k, 100 ms @24k) — same as the ElevenLabs integration.
        this.chunkSize = 4800;

        this.ws = null;
        this.wsReady = false;
        this.connectPromise = null;
        this.activeRequest = null; // { resolve, reject, buffer } for the in-flight utterance
        this.synthTimeoutMs = Number(process.env.VOX_TTS_TIMEOUT_MS || 30000);

        this.ready();
    }

    setVoice(voice) {
        this.voice = voice;
    }

    async ready() {
        // Establish the socket up front so the first utterance is fast.
        try {
            await this.ensureConnected();
        } catch (err) {
            logger.error(`${this.agentName} ${this.userName} VoxServe: initial connect failed: ${err.message}`);
        }
        this.emit('tts_ready', true);
    }

    ensureConnected() {
        if (this.ws && this.wsReady) return Promise.resolve();
        if (this.connectPromise) return this.connectPromise;

        this.connectPromise = new Promise((resolve, reject) => {
            const ws = new WebSocket(this.wsUrl);
            this.ws = ws;
            this.wsReady = false;

            const onOpenTimeout = setTimeout(() => {
                reject(new Error('VoxServe WS connect timeout'));
                try { ws.terminate(); } catch (_) {}
            }, this.synthTimeoutMs);

            ws.on('open', () => {
                clearTimeout(onOpenTimeout);
                this.wsReady = true;
                this.connectPromise = null;
                logger.info(`${this.agentName} ${this.userName} VoxServe WS connected: ${this.wsUrl}`);
                resolve();
            });

            ws.on('message', (data, isBinary) => this.handleMessage(data, isBinary));

            ws.on('error', (err) => {
                logger.error(`${this.agentName} ${this.userName} VoxServe WS error: ${err.message}`);
                if (!this.wsReady) {
                    clearTimeout(onOpenTimeout);
                    this.connectPromise = null;
                    reject(err);
                }
                this.failActiveRequest(err);
            });

            ws.on('close', () => {
                this.wsReady = false;
                this.ws = null;
                this.connectPromise = null;
                logger.info(`${this.agentName} ${this.userName} VoxServe WS closed`);
                this.failActiveRequest(new Error('VoxServe WS closed'));
            });
        });

        return this.connectPromise;
    }

    handleMessage(data, isBinary) {
        // Binary frames are raw int16 PCM (or Opus if requested — we use PCM).
        if (isBinary || Buffer.isBuffer(data)) {
            // Some ws versions deliver text frames as Buffer; disambiguate by
            // trying to parse control frames only when not flagged binary.
            if (!isBinary) {
                const asText = data.toString('utf8');
                if (asText.startsWith('{')) return this.handleControl(asText);
            }
            if (this.activeRequest) {
                this.emitPcm(Buffer.isBuffer(data) ? data : Buffer.from(data));
            }
            return;
        }
        this.handleControl(typeof data === 'string' ? data : data.toString('utf8'));
    }

    handleControl(text) {
        let msg;
        try {
            msg = JSON.parse(text);
        } catch (_) {
            return; // ignore non-JSON control frames
        }

        if (msg.type === 'start') {
            // New utterance stream beginning; nothing to do (buffer already reset).
            return;
        }
        if (msg.type === 'end') {
            this.finishActiveRequest();
            return;
        }
        if (msg.type === 'error') {
            logger.error(`${this.agentName} ${this.userName} VoxServe synth error: ${msg.detail}`);
            this.failActiveRequest(new Error(msg.detail || 'VoxServe synth error'));
            return;
        }
    }

    emitPcm(chunk) {
        const req = this.activeRequest;
        if (!req) return;
        req.buffer = Buffer.concat([req.buffer, chunk]);
        while (req.buffer.length >= this.chunkSize) {
            const chunkToEmit = req.buffer.subarray(0, this.chunkSize);
            this.emit('audio', chunkToEmit);
            req.buffer = req.buffer.subarray(this.chunkSize);
        }
    }

    finishActiveRequest() {
        const req = this.activeRequest;
        if (!req) return;
        // Flush the tail, padded with silence to a full chunk (matches EL).
        if (req.buffer.length > 0) {
            let finalChunk = req.buffer;
            if (finalChunk.length < this.chunkSize) {
                finalChunk = Buffer.concat([finalChunk, Buffer.alloc(this.chunkSize - finalChunk.length, 0)]);
            }
            this.emit('audio', finalChunk);
        }
        req.buffer = Buffer.alloc(0);
        clearTimeout(req.timer);
        this.activeRequest = null;
        req.resolve();
    }

    failActiveRequest(err) {
        const req = this.activeRequest;
        if (!req) return;
        clearTimeout(req.timer);
        this.activeRequest = null;
        req.reject(err);
    }

    async generate(gptReply, interactionId) {
        const { partialResponse } = gptReply;

        if (!partialResponse) {
            logger.error(`${this.agentName} ${this.userName} Invalid input or agent data not set.`);
            return;
        }

        if (this.currentInteractionId !== interactionId) {
            logger.info(`${this.agentName} ${this.userName} New interaction ${interactionId}, clearing queue`);
            this.messageQueue = [];
            this.isProcessing = false;
            this.currentInteractionId = interactionId;
        }

        this.messageQueue.push({
            text: partialResponse,
            processed: false,
            attempts: 0
        });

        if (!this.isProcessing) {
            await this.processNextInQueue();
        }
    }

    async processNextInQueue() {
        if (this.messageQueue.length === 0) {
            this.isProcessing = false;
            return;
        }

        this.isProcessing = true;
        const currentMessage = this.messageQueue[0];

        if (currentMessage.attempts >= 1) {
            logger.info(`${this.agentName} ${this.userName} Skipping message after ${currentMessage.attempts} attempts: ${currentMessage.text}`);
            this.messageQueue.shift();
            await this.processNextInQueue();
            return;
        }

        try {
            this.audioReceived = false;
            currentMessage.attempts++;
            await this.generateStreamingAudio(currentMessage.text);
        } catch (error) {
            logger.error('Error:', error);
        }
    }

    async generateStreamingAudio(text) {
        try {
            const timeStart = Date.now();
            logger.info(`${this.agentName} ${this.userName} TTS stream started`);

            await this.ensureConnected();

            await new Promise((resolve, reject) => {
                this.activeRequest = {
                    resolve,
                    reject,
                    buffer: Buffer.alloc(0),
                    timer: setTimeout(() => this.failActiveRequest(new Error('VoxServe synth timeout')), this.synthTimeoutMs),
                };

                const payload = {
                    text,
                    sample_rate: this.sampleRate,
                    format: 'pcm',
                };
                if (this.voice) payload.voice_id = this.voice;
                if (this.speakingRate !== undefined) payload.speaking_rate = this.speakingRate;
                if (this.pitchStd !== undefined) payload.pitch_std = this.pitchStd;

                try {
                    this.ws.send(JSON.stringify(payload));
                    logger.info(`${this.agentName} ${this.userName} TTS request sent: ${Date.now() - timeStart} ms`);
                } catch (err) {
                    this.failActiveRequest(err);
                }
            });

            logger.info(`${this.agentName} ${this.userName} Audio stream completed`);
            this.audioReceived = true;
            this.isConnecting = false;
            this.messageQueue.shift();
            this.processNextInQueue();
        } catch (error) {
            logger.error(`${this.agentName} ${this.userName} Error during TTS generation:`, error);
            this.isConnecting = false;
            // Drop the failed message so the queue can advance (mirrors attempts cap).
            if (this.messageQueue.length > 0) this.messageQueue.shift();
            this.processNextInQueue();
        }
    }

    stop() {
        this.isSpeaking = true;
        this.messageQueue = [];
        this.isProcessing = false;
    }

    end() {
        logger.info(`${this.agentName} ${this.userName} TTS Ended`);
        try {
            if (this.ws) {
                // Politely end the session, then close the socket.
                if (this.wsReady) this.ws.send(JSON.stringify({ type: 'close' }));
                this.ws.close();
            }
        } catch (_) {}
        this.ws = null;
        this.wsReady = false;
    }
}

module.exports = { TextToSpeechServiceVoxServe };
