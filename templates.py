"""
HTML Templates - Provides HTML interface for voice interaction
"""


def get_html_template() -> str:
    """
    Get HTML template for voice interface
    
    Returns:
        HTML content as string
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Appointment Voice Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .status.disconnected {
            background: #fee;
            color: #c33;
        }
        
        .status.connected {
            background: #efe;
            color: #3c3;
        }
        
        .status.listening {
            background: #fef3e0;
            color: #f59e0b;
        }
        
        .control-btn {
            width: 100%;
            padding: 18px;
            font-size: 18px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .control-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .start-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .stop-btn {
            background: #ef4444;
            color: white;
        }
        
        .transcript-box {
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 8px;
            line-height: 1.5;
        }
        
        .message.user {
            background: #e0e7ff;
            color: #3730a3;
            margin-left: 20px;
        }
        
        .message.assistant {
            background: #f3f4f6;
            color: #1f2937;
            margin-right: 20px;
        }
        
        .message.system {
            background: #fef3c7;
            color: #92400e;
            font-size: 14px;
            font-style: italic;
        }
        
        .rag-indicator {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin-left: 8px;
            font-weight: 600;
        }
        
        .features {
            background: #f9fafb;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .features h3 {
            color: #374151;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .features ul {
            list-style: none;
            font-size: 13px;
            color: #6b7280;
        }
        
        .features li {
            padding: 5px 0;
        }
        
        .features li:before {
            content: "✓ ";
            color: #10b981;
            font-weight: bold;
            margin-right: 5px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .listening-indicator {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Medical Appointment Assistant</h1>
        <p class="subtitle">AI-powered voice scheduling with medical knowledge</p>
        
        <div id="status" class="status disconnected">
            🔴 Disconnected
        </div>
        
        <button id="startBtn" class="control-btn start-btn">
            🎤 Start Voice Conversation
        </button>
        
        <button id="stopBtn" class="control-btn stop-btn" style="display:none;">
            🛑 Stop Conversation
        </button>
        
        <div class="transcript-box" id="transcript">
            <div class="message system">
                Click "Start Voice Conversation" to begin. Make sure your microphone is enabled.
            </div>
        </div>
        
        <div class="features">
            <h3>Features:</h3>
            <ul>
                <li>Natural voice conversation</li>
                <li>Medical knowledge integration (RAG)</li>
                <li>Emergency detection</li>
                <li>Specialty recommendations</li>
                <li>Appointment scheduling</li>
            </ul>
        </div>
    </div>
    
    <audio id="audioPlayer" style="display:none;"></audio>
    
    <script>
        let websocket = null;
        let mediaRecorder = null;
        let audioContext = null;
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const transcript = document.getElementById('transcript');
        const audioPlayer = document.getElementById('audioPlayer');
        
        function updateStatus(text, className) {
            status.textContent = text;
            status.className = 'status ' + className;
        }
        
        function addMessage(text, type) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + type;
            msgDiv.textContent = text;
            transcript.appendChild(msgDiv);
            transcript.scrollTop = transcript.scrollHeight;
        }
        
        async function startConversation() {
            try {
                updateStatus('🟡 Connecting...', 'listening');
                
                // Request microphone permission
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Create WebSocket connection
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/voice`;
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {
                    updateStatus('🟢 Connected - Listening...', 'connected');
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'block';
                    
                    // Start recording
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm'
                    });
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket.readyState === WebSocket.OPEN) {
                            websocket.send(event.data);
                        }
                    };
                    
                    mediaRecorder.start(250); // Send chunks every 250ms
                };
                
                websocket.onmessage = async (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'transcript') {
                        addMessage(data.text, 'user');
                    } else if (data.type === 'response') {
                        let responseText = data.text;
                        if (data.used_rag) {
                            responseText += ' <span class="rag-indicator">RAG</span>';
                        }
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message assistant';
                        msgDiv.innerHTML = responseText;
                        transcript.appendChild(msgDiv);
                        transcript.scrollTop = transcript.scrollHeight;
                    } else if (data.type === 'audio') {
                        // Play audio response
                        const audioBlob = base64ToBlob(data.audio, 'audio/mpeg');
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayer.src = audioUrl;
                        audioPlayer.play();
                    } else if (data.type === 'error') {
                        addMessage('Error: ' + data.message, 'system');
                    } else if (data.type === 'rag_status' && data.using_rag) {
                        updateStatus('🟢 Using medical knowledge...', 'connected listening-indicator');
                        setTimeout(() => {
                            updateStatus('🟢 Connected - Listening...', 'connected');
                        }, 2000);
                    }
                };
                
                websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    addMessage('Connection error occurred', 'system');
                    updateStatus('🔴 Error', 'disconnected');
                };
                
                websocket.onclose = () => {
                    updateStatus('🔴 Disconnected', 'disconnected');
                    stopConversation();
                };
                
            } catch (error) {
                console.error('Error starting conversation:', error);
                addMessage('Failed to start: ' + error.message, 'system');
                updateStatus('🔴 Failed to start', 'disconnected');
            }
        }
        
        function stopConversation() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            if (websocket) {
                websocket.close();
                websocket = null;
            }
            
            startBtn.style.display = 'block';
            stopBtn.style.display = 'none';
            updateStatus('🔴 Disconnected', 'disconnected');
        }
        
        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }
        
        startBtn.addEventListener('click', startConversation);
        stopBtn.addEventListener('click', stopConversation);
    </script>
</body>
</html>
    """