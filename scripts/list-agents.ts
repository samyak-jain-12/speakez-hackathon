
const API_KEY = "4gakmRfS2PgXjVadcU3v5HZVcS6jDy8E";
const BASE_URL = "https://api.on-demand.io/chat/v1";

async function listAgents() {
    console.log("Attempting to list agents...");
    // Try common endpoints
    const endpoints = [
        `${BASE_URL}/agents`,
        `https://api.on-demand.io/registry/v1/agents`, // Guessing registry URL
        `https://api.on-demand.io/agents`
    ];

    for (const url of endpoints) {
        try {
            console.log(`Trying ${url}...`);
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'apikey': API_KEY,
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                console.log(`SUCCESS calling ${url}`);
                console.log(JSON.stringify(data, null, 2));
                return;
            } else {
                console.log(`Failed calling ${url}: ${response.status} ${response.statusText}`);
                const text = await response.text();
                console.log("Response:", text);
            }
        } catch (err) {
            // console.error(`Error calling ${url}:`, err.message);
            console.error('Error')
        }
    }
}

listAgents();
