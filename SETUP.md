# SPEAKEZ Setup Guide

## Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```env
# OpenAI API Key for GPT-powered roadmap generation
OPENAI_API_KEY=your_openai_api_key_here

# YouTube Data API Key for learning resources
YOUTUBE_API_KEY=your_youtube_api_key_here

# App URL
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

## Installation

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables (see above)

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Features Implemented

✅ Authentication (Landing, Signup, Login, Forgot Password)
✅ Dashboard with sidebar navigation
✅ Practice Session with microphone/camera support
✅ Real-time metrics display
✅ Multi-Agent Architecture (6 agents + orchestrator)
✅ Purpose-based Speech Modes
✅ Scenario Simulator
✅ Post-Practice Insights
✅ Improvement Roadmap (GPT-powered)
✅ YouTube Learning Resources
✅ Profile & Settings with Accent Mode
✅ Speech-to-Text API integration (ready for production)

## Architecture

### Multi-Agent System
- **Speech Signal Agent**: Analyzes pitch, volume, pace, filler words
- **Fluency Agent**: Detects stuttering, repetition, pauses
- **Stress & Emotion Agent**: Analyzes stress levels and emotions
- **Body Language Agent**: Analyzes posture, eye contact, gestures
- **Context Mode Agent**: Adapts feedback based on selected mode
- **Roadmap Agent**: Generates personalized improvement plans
- **Orchestrator Agent**: Coordinates all agents

### API Routes
- `/api/roadmap` - GPT-powered roadmap generation
- `/api/youtube` - YouTube video recommendations
- `/api/speech-to-text` - Speech transcription (ready for production APIs)

## Philosophy

SPEAKEZ is built on principles of:
- Non-judgmental support
- Personal baseline adaptation
- Inclusive design
- Accessibility-first approach
- Supportive, encouraging language

## Next Steps for Production

1. Set up proper authentication (NextAuth.js, Auth0, etc.)
2. Integrate real Speech-to-Text API (Google Cloud, AWS, Azure)
3. Set up database (PostgreSQL, MongoDB, etc.)
4. Deploy to production (Vercel, AWS, etc.)
5. Add comprehensive error handling
6. Implement proper session management
7. Add analytics and monitoring
