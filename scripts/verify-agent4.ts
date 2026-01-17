
import { processAgent4Video } from '../lib/agents/agent4';
import fs from 'fs';
import path from 'path';

async function verify() {
    console.log("Starting verification of Agent 4...");

    // Create a minimal dummy video file (just text content, may fail if API checks magic numbers, but usually APIs just upload)
    // Ideally we would use a real small video file, but for a quick check let's try a buffer.
    // Actually, let's look for a dummy video or create a very simple file.
    // If the API expects a real video format, this might fail with "invalid file".
    // But let's try to pass a Buffer which mocks a Blob.

    const dummyContent = Buffer.from("dummy video content");

    // Mock Blob since Node.js doesn't have Blob globally in older versions, but recent Node has.
    // We'll see if the environment supports it. 
    // If not, we might need a polyfill or just cast it.

    const blob = new Blob([dummyContent], { type: 'video/webm' });

    try {
        console.log("Calling processAgent4Video...");
        const result = await processAgent4Video({
            file: blob,
            fileName: "test_verification_video.webm",
            sessionId: undefined // Should create new session
        });

        console.log("Verification SUCCESS!");
        console.log("Result:", JSON.stringify(result, null, 2));
    } catch (error) {
        console.error("Verification FAILED:");
        console.error(error);
    }
}

verify();
