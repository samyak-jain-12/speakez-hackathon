export class RoadmapAgent {
  async generate(sessionData: any, userProfile: any): Promise<string> {
    // In production, this would call GPT API
    // For now, generate a structured roadmap based on session data
    
    const patterns = this.identifyPatterns(sessionData)
    const roadmap = this.buildRoadmap(patterns, userProfile)
    
    return roadmap
  }

  private identifyPatterns(sessionData: any): string[] {
    const patterns: string[] = []
    
    if (sessionData.metrics?.pace?.length > 0) {
      const avgPace = sessionData.metrics.pace.reduce((a: number, b: number) => a + b, 0) / sessionData.metrics.pace.length
      if (avgPace > 180) {
        patterns.push('rapid-speech')
      } else if (avgPace < 120) {
        patterns.push('slow-speech')
      }
    }
    
    if (sessionData.metrics?.stress?.some((s: number) => s > 0.6)) {
      patterns.push('high-stress')
    }
    
    if (sessionData.metrics?.fillerWords?.reduce((a: number, b: number) => a + b, 0) > 10) {
      patterns.push('filler-words')
    }
    
    if (sessionData.metrics?.stuttering?.some((s: number) => s > 0.3)) {
      patterns.push('stuttering')
    }
    
    return patterns
  }

  private buildRoadmap(patterns: string[], userProfile: any): string {
    let roadmap = `# Personalized Improvement Roadmap\n\n`
    roadmap += `Based on your practice session, here's a step-by-step plan:\n\n`
    
    if (patterns.includes('rapid-speech')) {
      roadmap += `## Day 1-3: Pacing Practice\n`
      roadmap += `- Practice reading aloud at 140 words per minute\n`
      roadmap += `- Use a metronome app to maintain steady pace\n`
      roadmap += `- Focus on pausing after each sentence\n\n`
    }
    
    if (patterns.includes('high-stress')) {
      roadmap += `## Day 4-7: Stress Management\n`
      roadmap += `- Practice breathing exercises before speaking\n`
      roadmap += `- Record yourself in low-pressure scenarios first\n`
      roadmap += `- Gradually increase complexity of practice scenarios\n\n`
    }
    
    if (patterns.includes('filler-words')) {
      roadmap += `## Day 8-10: Reducing Filler Words\n`
      roadmap += `- Practice pausing instead of using filler words\n`
      roadmap += `- Record yourself and identify filler word patterns\n`
      roadmap += `- Use structured thinking: pause, think, then speak\n\n`
    }
    
    if (patterns.includes('stuttering')) {
      roadmap += `## Ongoing: Fluency Support\n`
      roadmap += `- This is common and manageable with practice\n`
      roadmap += `- Practice at your own comfortable pace\n`
      roadmap += `- Focus on message clarity over speed\n`
      roadmap += `- Consider working with a speech therapist for additional support\n\n`
    }
    
    if (patterns.length === 0) {
      roadmap += `## Continue Your Practice\n`
      roadmap += `- Your speech patterns look strong!\n`
      roadmap += `- Keep practicing to maintain consistency\n`
      roadmap += `- Try different scenarios to build versatility\n\n`
    }
    
    roadmap += `## General Tips\n`
    roadmap += `- Practice 10-15 minutes daily\n`
    roadmap += `- Record yourself regularly to track progress\n`
    roadmap += `- Be patient with yourself - improvement takes time\n`
    roadmap += `- Celebrate small wins along the way\n\n`
    
    if (userProfile?.hasSpeechDisorder) {
      roadmap += `## Personalized Note\n`
      roadmap += `Remember, your speech patterns are unique. This roadmap adapts to your personal baseline, not a universal standard. Take your time and practice at a pace that feels comfortable for you.\n\n`
    }
    
    return roadmap
  }
}
