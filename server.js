const express = require('express');
const path = require('path');
const fetch = require('node-fetch');
const fs = require('fs');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(__dirname));
app.use(express.json());

// Load API key
let openaiApiKey = '';
try {
    openaiApiKey = fs.readFileSync('OPENAI_API_KEY.txt', 'utf8').trim();
} catch (error) {
    console.log('No API key file found, using environment variable or manual input');
}

// OpenAI API proxy endpoint
app.post('/api/openai', async (req, res) => {
    try {
        const { prompt, model = 'gpt-3.5-turbo', max_tokens = 2000 } = req.body;
        
        if (!prompt) {
            return res.status(400).json({ error: 'Prompt is required' });
        }

        if (!openaiApiKey) {
            return res.status(500).json({ error: 'OpenAI API key not configured' });
        }

        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${openaiApiKey}`
            },
            body: JSON.stringify({
                model: model,
                messages: [
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                max_tokens: max_tokens,
                temperature: 0.7
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            return res.status(response.status).json({ 
                error: data.error?.message || 'OpenAI API error' 
            });
        }

        res.json({
            success: true,
            response: data.choices[0].message.content,
            usage: data.usage
        });

    } catch (error) {
        console.error('OpenAI API error:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            details: error.message 
        });
    }
});

// Get API key status
app.get('/api/status', (req, res) => {
    res.json({
        apiKeyConfigured: !!openaiApiKey,
        apiKeyLength: openaiApiKey ? openaiApiKey.length : 0
    });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`API key configured: ${!!openaiApiKey}`);
});
