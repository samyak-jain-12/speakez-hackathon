import fs from 'fs';

async function verifyServer() {
    const url = 'http://localhost:3002/api/agent/video';
    console.log(`Calling ${url}...`);

    try {
        // Create a dummy file buffer
        const buffer = Buffer.from("dummy video content");
        const blob = new Blob([buffer], { type: 'video/webm' });

        const formData = new FormData();
        formData.append('file', blob, 'test.webm');


        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            console.log("Server Verification SUCCESS!");
            console.log(JSON.stringify(data, null, 2));
        } else {
            console.error(`Server Verification FAILED: ${response.status} ${response.statusText}`);
            const text = await response.text();
            console.error(text);
        }
    } catch (err) {
        console.error("Error calling server:", err);
    }
}

verifyServer();
