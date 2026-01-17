import { SpeechSignalAgent } from './speech-signal'
import { FluencyAgent } from './fluency'
import { StressEmotionAgent } from './stress-emotion'
import { BodyLanguageAgent } from './body-language'
import { ContextModeAgent } from './context-mode'
import { RoadmapAgent } from './roadmap'

export interface AgentResult {
  agent: string
  data: any
  timestamp: number
}

export class OrchestratorAgent {
  private speechSignal: SpeechSignalAgent
  private fluency: FluencyAgent
  private stressEmotion: StressEmotionAgent
  private bodyLanguage: BodyLanguageAgent
  private contextMode: ContextModeAgent
  private roadmap: RoadmapAgent

  constructor() {
    this.speechSignal = new SpeechSignalAgent()
    this.fluency = new FluencyAgent()
    this.stressEmotion = new StressEmotionAgent()
    this.bodyLanguage = new BodyLanguageAgent()
    this.contextMode = new ContextModeAgent()
    this.roadmap = new RoadmapAgent()
  }

  async processAudio(audioData: Float32Array, mode: string): Promise<AgentResult[]> {
    const results: AgentResult[] = []

    // Activate speech signal agent (always active)
    const speechResult = await this.speechSignal.analyze(audioData)
    results.push({
      agent: 'speech-signal',
      data: speechResult,
      timestamp: Date.now(),
    })

    // Activate fluency agent if needed
    if (speechResult.pace > 200 || speechResult.fillerWords > 5) {
      const fluencyResult = await this.fluency.analyze(audioData, speechResult)
      results.push({
        agent: 'fluency',
        data: fluencyResult,
        timestamp: Date.now(),
      })
    }

    // Activate stress/emotion agent
    if (speechResult.stress > 0.5 || speechResult.volume > 0.8) {
      const stressResult = await this.stressEmotion.analyze(audioData, speechResult)
      results.push({
        agent: 'stress-emotion',
        data: stressResult,
        timestamp: Date.now(),
      })
    }

    // Context mode agent (always active for mode-specific adjustments)
    const contextResult = await this.contextMode.analyze(speechResult, mode)
    results.push({
      agent: 'context-mode',
      data: contextResult,
      timestamp: Date.now(),
    })

    return results
  }

  async processVideo(videoFrame: ImageData): Promise<AgentResult> {
    const bodyResult = await this.bodyLanguage.analyze(videoFrame)
    return {
      agent: 'body-language',
      data: bodyResult,
      timestamp: Date.now(),
    }
  }

  async generateRoadmap(sessionData: any, userProfile: any): Promise<string> {
    return await this.roadmap.generate(sessionData, userProfile)
  }
}
