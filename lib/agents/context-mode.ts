export class ContextModeAgent {
  private modeConfigs: Record<string, any> = {
    pitch: {
      idealPace: { min: 140, max: 160 },
      gestureTolerance: 'high',
      toneExpectation: 'enthusiastic',
      feedbackLanguage: 'energetic',
    },
    interview: {
      idealPace: { min: 120, max: 150 },
      gestureTolerance: 'medium',
      toneExpectation: 'professional',
      feedbackLanguage: 'formal',
    },
    professional: {
      idealPace: { min: 130, max: 160 },
      gestureTolerance: 'low',
      toneExpectation: 'calm',
      feedbackLanguage: 'professional',
    },
    debate: {
      idealPace: { min: 150, max: 180 },
      gestureTolerance: 'high',
      toneExpectation: 'assertive',
      feedbackLanguage: 'confident',
    },
    conversation: {
      idealPace: { min: 120, max: 180 },
      gestureTolerance: 'high',
      toneExpectation: 'natural',
      feedbackLanguage: 'casual',
    },
  }

  async analyze(speechSignal: any, mode: string): Promise<{
    mode: string
    paceFeedback: string
    toneFeedback: string
    suggestions: string[]
  }> {
    const config = this.modeConfigs[mode] || this.modeConfigs.conversation
    
    // Analyze pace against mode expectations
    const paceFeedback = this.analyzePace(speechSignal.pace, config.idealPace)
    
    // Analyze tone
    const toneFeedback = this.analyzeTone(speechSignal, config.toneExpectation)
    
    // Generate mode-specific suggestions
    const suggestions = this.generateSuggestions(speechSignal, config, mode)

    return {
      mode,
      paceFeedback,
      toneFeedback,
      suggestions,
    }
  }

  private analyzePace(currentPace: number, idealRange: { min: number; max: number }): string {
    if (currentPace < idealRange.min) {
      return `Your pace is slower than typical for this mode. Consider speaking slightly faster.`
    } else if (currentPace > idealRange.max) {
      return `Your pace is faster than typical for this mode. Consider slowing down slightly.`
    } else {
      return `Your pace fits well with this mode. Great job!`
    }
  }

  private analyzeTone(speechSignal: any, expectedTone: string): string {
    const currentTone = speechSignal.stress > 0.6 ? 'anxious' : 
                        speechSignal.volume > 0.7 ? 'enthusiastic' : 'calm'
    
    if (expectedTone === 'enthusiastic' && currentTone !== 'enthusiastic') {
      return `For this mode, try to bring more energy to your delivery.`
    } else if (expectedTone === 'professional' && currentTone === 'anxious') {
      return `Take a moment to breathe. Professional settings benefit from calm delivery.`
    } else {
      return `Your tone matches well with this mode.`
    }
  }

  private generateSuggestions(speechSignal: any, config: any, mode: string): string[] {
    const suggestions: string[] = []
    
    if (speechSignal.pace > config.idealPace.max) {
      suggestions.push(`For ${mode} mode, try pausing more between key points.`)
    }
    
    if (config.gestureTolerance === 'high' && speechSignal.fillerWords > 3) {
      suggestions.push(`Natural gestures can help reduce filler words.`)
    }
    
    if (mode === 'interview' && speechSignal.stress > 0.5) {
      suggestions.push(`Remember to breathe. Interviews are conversations, not tests.`)
    }
    
    return suggestions
  }
}
