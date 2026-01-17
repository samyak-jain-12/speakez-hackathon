import { NextRequest, NextResponse } from 'next/server'
import { processAgent5 } from '@/lib/agents/agent5'
import { processAgent6 } from '@/lib/agents/agent6'
import { processAgent1_5 } from '@/lib/agents/agent1_5'
import { processAgent3 } from '@/lib/agents/agent3'

const fs = require('fs');
const path = require('path');

function logDebug(message: string) {
    const logPath = path.join(process.cwd(), 'debug_analysis.log');
    fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${message}\n`);
}

export async function POST(req: NextRequest) {
    try {
        logDebug("Received POST request to /api/agent/analyze");
        const body = await req.json()
        const { agent2, agent4, mode, transcript, audioBlob } = body
        // Note: 'agent3' input is now computed here, not passed in from client mock

        logDebug(`Input Mode: ${mode}`);
        logDebug(`Transcript Length: ${transcript?.length || 0}`);
        logDebug(`Agent 2 Data: ${JSON.stringify(agent2)}`);

        // --- 0. Agent 1.5: Audio Storage (Optional) ---
        let audioUrl = null;
        // In a real scenario, we'd expect 'file' or 'audioBlob' to be handled. 
        // For this route, if we receive a base64 or blob, we'd pass it.
        // Assuming current frontend implementation might not send 'audioBlob' yet, skipping if null.

        // --- 1. Agent 3: NLP Analysis ---
        logDebug("Running Agent 3 (NLP)...");
        let agent3Data = null;
        try {
            const agent3Result = await processAgent3({
                transcript: transcript || "No transcript available",
                audioUrl: audioUrl || undefined
            });
            logDebug(`Agent 3 Raw Result: ${JSON.stringify(agent3Result)}`);

            agent3Data = agent3Result.data?.answer
                ? JSON.parse(agent3Result.data.answer.replace(/```json/g, '').replace(/```/g, ''))
                : agent3Result.data;

            // Robust parsing
            if (typeof agent3Data === 'string') {
                try { agent3Data = JSON.parse(agent3Data); } catch (e) { }
            }
        } catch (e: any) {
            logDebug(`Agent 3 Failed: ${e.message}`);
            // Fallback mock if agent 3 fails
            agent3Data = { filler_words: 0, clarity_score: 0.8, key_phrases_repeated: [] };
        }

        // --- Robust Agent 4 Data Handling (Mocking if missing) ---
        let agent4Data = agent4
        if (!agent4Data) {
            logDebug("Agent 4 Data missing, generating MOCK data...");
            const stress = agent2?.stress_index || 0;
            const pace = agent2?.pace || 130;

            let mockPosture = false; // "Slouching"
            let mockGestures = 'medium';
            let mockEyeContact = 0.75;

            // 1. Posture Calculation
            if (stress > 0.6) {
                mockPosture = true;
            }

            // 2. Gesture Intensity
            if (pace > 160) {
                mockGestures = 'high';
            } else if (pace < 110 || stress > 0.7) {
                mockGestures = 'low';
            }

            // 3. Eye Contact
            if (stress < 0.3) {
                mockEyeContact = 0.85;
            } else if (stress > 0.7) {
                mockEyeContact = 0.45;
            }

            agent4Data = {
                eye_contact_score: mockEyeContact,
                posture_alert: mockPosture,
                gesture_intensity: mockGestures
            }
        }

        logDebug(`Using Agent 4 Data: ${JSON.stringify(agent4Data)}`);
        logDebug(`Using Agent 3 Data: ${JSON.stringify(agent3Data)}`);

        // --- 2. Run Agent 5 (Confidence) ---
        // Agent 5 expects { agent2, agent3, agent4 }
        const agent5Input = {
            agent2,
            agent3: agent3Data, // Using real Agent 3 data
            agent4: agent4Data
        }

        logDebug("Running Agent 5 (Confidence)...")
        const agent5Result = await processAgent5(agent5Input)
        logDebug(`Agent 5 Raw Result: ${JSON.stringify(agent5Result)}`)

        let confidenceMetrics = agent5Result.data?.answer
            ? JSON.parse(agent5Result.data.answer.replace(/```json/g, '').replace(/```/g, ''))
            : agent5Result.data

        // Robust parsing check
        if (typeof confidenceMetrics === 'string') {
            try { confidenceMetrics = JSON.parse(confidenceMetrics); } catch (e) { }
        }
        if (!confidenceMetrics) confidenceMetrics = agent5Result // Fallback

        // --- HARDCODE OVERRIDE (User Request: Random 65-87) ---
        // Forces all scores to look "Professional" or "Good"
        if (confidenceMetrics) {
            const randomScore = () => Math.floor(Math.random() * (87 - 65 + 1)) + 65;

            confidenceMetrics.confidence_score = randomScore();

            if (confidenceMetrics.factors && Array.isArray(confidenceMetrics.factors)) {
                confidenceMetrics.factors.forEach((f: any) => {
                    f.score = randomScore();
                });
            } else {
                // Ensure factors exist if Agent 5 missed them
                confidenceMetrics.factors = [
                    { name: "Pace Control", score: randomScore(), impact: 20 },
                    { name: "Volume Consistency", score: randomScore(), impact: 20 },
                    { name: "Eye Contact", score: randomScore(), impact: 20 },
                    { name: "Stress Management", score: randomScore(), impact: 20 },
                    { name: "Clarity", score: randomScore(), impact: 20 }
                ];
            }
        }

        // --- 3. Run Agent 6 (Alignment) ---
        const agent6Input = {
            agent2,
            agent3: agent3Data,
            agent4: agent4Data,
            agent5: confidenceMetrics,
            mode: mode || 'Interview',
            transcript: transcript
        }

        logDebug("Running Agent 6 (Alignment) with Transcript...")
        const agent6Result = await processAgent6(agent6Input)
        logDebug(`Agent 6 Raw Result: ${JSON.stringify(agent6Result)}`)

        let alignmentMetrics = agent6Result.data?.answer
            ? JSON.parse(agent6Result.data.answer.replace(/```json/g, '').replace(/```/g, ''))
            : agent6Result.data

        if (typeof alignmentMetrics === 'string') {
            try { alignmentMetrics = JSON.parse(alignmentMetrics); } catch (e) { }
        }
        if (!alignmentMetrics) alignmentMetrics = agent6Result

        // --- 4. Run Agent 8 (Drill Generator) ---
        const weakPoints = []
        if (confidenceMetrics?.dominant_weakness) weakPoints.push(confidenceMetrics.dominant_weakness)
        if (!alignmentMetrics?.tone_match) weakPoints.push("Tone Alignment")
        // Add Agent 3 signal to weak points
        if (agent3Data?.clarity_score < 0.6) weakPoints.push("Clarity/Articulation")

        const agent8Input = {
            session_summary: `User scored ${confidenceMetrics?.confidence_score || 0}% confidence. Mode: ${mode}. Suggestion: ${alignmentMetrics?.suggestion || ''}`,
            weak_points: weakPoints
        }

        logDebug("Running Agent 8 (Drills)...")
        let drillMetrics = null
        try {
            const { processAgent8 } = require('@/lib/agents/agent8')
            const agent8Result = await processAgent8(agent8Input)
            logDebug(`Agent 8 Raw Result: ${JSON.stringify(agent8Result)}`)

            drillMetrics = agent8Result.data?.answer
                ? JSON.parse(agent8Result.data.answer.replace(/```json/g, '').replace(/```/g, ''))
                : agent8Result.data
        } catch (e: any) {
            logDebug(`Agent 8 Failed (Optional): ${e.message}`)
        }

        return NextResponse.json({
            confidenceMetrics,
            alignmentMetrics,
            drillMetrics
        })

    } catch (error: any) {
        logDebug(`Analysis Error: ${error.message}`);
        console.error("Analysis Error:", error)
        return NextResponse.json({ error: error.message }, { status: 500 })
    }
}
