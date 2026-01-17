export class FluencyAgent {
  private sampleRate: number;

  constructor(sampleRate = 16000) {
    this.sampleRate = sampleRate;
  }

  async analyze(
    audioData: Float32Array,
    speechSignal?: any,
  ): Promise<{
    stuttering: number;
    repetition: number;
    pauses: number;
    isDisorderPattern: boolean;
    reassurance: string;
  }> {
    if (!audioData || audioData.length === 0) {
      return {
        stuttering: 0,
        repetition: 0,
        pauses: 0,
        isDisorderPattern: false,
        reassurance: "No audio detected. Please try again.",
      };
    }

    // 1) Preprocess: normalize + remove DC offset
    const clean = this.normalize(this.removeDC(audioData));

    // 2) Feature extraction
    const frames = this.frameSignal(clean, 0.02, 0.01); // 20ms frames, 10ms hop
    const frameEnergy = frames.map((f) => this.rms(f));
    const frameZCR = frames.map((f) => this.zeroCrossingRate(f));

    // 3) Pause detection (based on duration)
    const pauses = this.countPausesFromFrames(frameEnergy);

    // 4) Stuttering score (instability in voiced regions)
    const stuttering = this.detectStutteringFromInstability(
      frameEnergy,
      frameZCR,
    );

    // 5) Repetition score (repeated voiced segments pattern)
    const repetition = this.detectRepetitionFromVoicedBursts(frameEnergy);

    // Disorder heuristic (tune thresholds later)
    const isDisorderPattern = stuttering > 0.45 || repetition > 0.45;

    const reassurance = this.generateReassurance(
      stuttering,
      repetition,
      pauses,
    );

    return {
      stuttering,
      repetition,
      pauses,
      isDisorderPattern,
      reassurance,
    };
  }

  // ---------------------------
  // Core detectors
  // ---------------------------

  private detectStutteringFromInstability(
    frameEnergy: number[],
    frameZCR: number[],
  ): number {
    // voiced frames = energy above a dynamic threshold
    const threshold = this.percentile(frameEnergy, 60) * 0.6;
    const voicedIdx: number[] = [];

    for (let i = 0; i < frameEnergy.length; i++) {
      if (frameEnergy[i] > threshold) voicedIdx.push(i);
    }

    if (voicedIdx.length < 10) return 0;

    // compute instability: large frame-to-frame deltas in energy + zcr
    let unstableCount = 0;
    let totalPairs = 0;

    for (let k = 1; k < voicedIdx.length; k++) {
      const i1 = voicedIdx[k - 1];
      const i2 = voicedIdx[k];

      // skip big gaps (not consecutive voiced)
      if (i2 - i1 > 2) continue;

      const dE = Math.abs(frameEnergy[i2] - frameEnergy[i1]);
      const dZ = Math.abs(frameZCR[i2] - frameZCR[i1]);

      // tuned conservative thresholds
      if (dE > 0.04 || dZ > 0.12) unstableCount++;
      totalPairs++;
    }

    if (totalPairs === 0) return 0;

    // score 0..1
    return this.clamp01(unstableCount / totalPairs);
  }

  private detectRepetitionFromVoicedBursts(frameEnergy: number[]): number {
    // We split speech into bursts (voiced segments separated by silence)
    const threshold = this.percentile(frameEnergy, 60) * 0.5;

    const bursts: Array<{ start: number; end: number; avgEnergy: number }> = [];
    let start = -1;

    for (let i = 0; i < frameEnergy.length; i++) {
      const voiced = frameEnergy[i] > threshold;

      if (voiced && start === -1) start = i;
      if (!voiced && start !== -1) {
        const end = i - 1;
        const avg = this.mean(frameEnergy.slice(start, end + 1));
        bursts.push({ start, end, avgEnergy: avg });
        start = -1;
      }
    }

    if (start !== -1) {
      const end = frameEnergy.length - 1;
      const avg = this.mean(frameEnergy.slice(start, end + 1));
      bursts.push({ start, end, avgEnergy: avg });
    }

    if (bursts.length < 3) return 0;

    // repetition heuristic:
    // if many bursts have very similar duration and avgEnergy -> likely repeated short chunks
    let repeats = 0;
    let comparisons = 0;

    for (let i = 1; i < bursts.length; i++) {
      const b1 = bursts[i - 1];
      const b2 = bursts[i];

      const dur1 = b1.end - b1.start;
      const dur2 = b2.end - b2.start;

      const durSim = 1 - Math.min(1, Math.abs(dur1 - dur2) / Math.max(1, dur1));
      const eSim =
        1 -
        Math.min(
          1,
          Math.abs(b1.avgEnergy - b2.avgEnergy) / Math.max(1e-6, b1.avgEnergy),
        );

      if (durSim > 0.7 && eSim > 0.75) repeats++;
      comparisons++;
    }

    if (comparisons === 0) return 0;
    return this.clamp01(repeats / comparisons);
  }

  private countPausesFromFrames(frameEnergy: number[]): number {
    // pause = consecutive low-energy frames lasting >= 250ms
    const threshold = this.percentile(frameEnergy, 40) * 0.5;
    const frameHopSec = 0.01; // hop = 10ms
    const minPauseSec = 0.25;

    let pauses = 0;
    let silentRun = 0;

    for (let i = 0; i < frameEnergy.length; i++) {
      const silent = frameEnergy[i] < threshold;

      if (silent) silentRun++;
      else {
        const silentSec = silentRun * frameHopSec;
        if (silentSec >= minPauseSec) pauses++;
        silentRun = 0;
      }
    }

    // trailing silence
    const tailSec = silentRun * frameHopSec;
    if (tailSec >= minPauseSec) pauses++;

    return pauses;
  }

  // ---------------------------
  // Helpers
  // ---------------------------

  private frameSignal(signal: Float32Array, frameSec: number, hopSec: number) {
    const frameLen = Math.max(1, Math.floor(frameSec * this.sampleRate));
    const hopLen = Math.max(1, Math.floor(hopSec * this.sampleRate));

    const frames: Float32Array[] = [];
    for (let start = 0; start + frameLen <= signal.length; start += hopLen) {
      frames.push(signal.slice(start, start + frameLen));
    }
    return frames;
  }

  private rms(frame: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    return Math.sqrt(sum / frame.length);
  }

  private zeroCrossingRate(frame: Float32Array): number {
    let crossings = 0;
    for (let i = 1; i < frame.length; i++) {
      const prev = frame[i - 1];
      const curr = frame[i];
      if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) crossings++;
    }
    return crossings / frame.length;
  }

  private removeDC(x: Float32Array): Float32Array {
    const mean = this.mean(x);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] - mean;
    return out;
  }

  private normalize(x: Float32Array): Float32Array {
    let maxAbs = 0;
    for (let i = 0; i < x.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(x[i]));
    }
    if (maxAbs < 1e-9) return x;

    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] / maxAbs;
    return out;
  }

  private mean(arr: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return arr.length ? s / arr.length : 0;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.floor((p / 100) * (sorted.length - 1));
    return sorted[idx];
  }

  private clamp01(x: number): number {
    return Math.max(0, Math.min(1, x));
  }

  private generateReassurance(
    stuttering: number,
    repetition: number,
    pauses: number,
  ): string {
    if (stuttering > 0.45 || repetition > 0.45) {
      return "It’s okay—speech disruptions are common. Focus on slowing down and breathing. You’re improving with practice.";
    }
    if (pauses >= 3) {
      return "Pauses are normal and can even improve clarity. Take your time—you're doing great.";
    }
    return "Your speech flow looks smooth. Keep practicing!";
  }
}
