import { useEffect, useRef, useState } from 'react'
import { signOut } from 'firebase/auth'
import { auth } from '../firebase'
import MessageBubble from '../components/MessageBubble'
import DebugPanel from '../components/DebugPanel'

export default function Chat({ user, onOpenHistory }) {
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [streamingText, setStreamingText] = useState('')
  const [debugMode, setDebugMode] = useState(false)
  const [debugEvents, setDebugEvents] = useState([])
  const messagesEndRef = useRef(null)
  const streamingTextRef = useRef('')

  const getToken = () => user.getIdToken()

  const createSession = async () => {
    try {
      const token = await getToken()
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages([])
      setDebugEvents([])
    } catch (e) {
      console.error('Failed to create session:', e)
    }
  }

  useEffect(() => { createSession() }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const userMessage = input.trim()
    if (!userMessage || isLoading || !sessionId) return

    setInput('')
    streamingTextRef.current = ''
    setStreamingText('')
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)

    try {
      const token = await getToken()
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ user_input: userMessage, session_id: sessionId, debug: debugMode }),
      })

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep any incomplete trailing line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let event
          try {
            event = JSON.parse(line.slice(6))
          } catch {
            continue
          }

          if (event.type === 'status') {
            setStatusMessage(event.message)
          } else if (event.type === 'token') {
            // streaming tokens (only if using LangChain ChatModel)
            streamingTextRef.current += event.content
            setStreamingText(streamingTextRef.current)
          } else if (event.type === 'message_end') {
            // streaming path: commit accumulated tokens
            const text = streamingTextRef.current
            streamingTextRef.current = ''
            setStreamingText('')
            if (text) setMessages((prev) => [...prev, { role: 'assistant', content: text }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'message') {
            // non-streaming path: full post delivered at once
            streamingTextRef.current = ''
            setStreamingText('')
            setMessages((prev) => [...prev, { role: 'assistant', content: event.content }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'debug') {
            setDebugEvents((prev) => [...prev, { node: event.node, payload: event.payload, ts: Date.now() }])
          } else if (event.type === 'clarification') {
            setMessages((prev) => [...prev, { role: 'assistant', content: event.payload.question }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'done') {
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'error') {
            setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${event.payload.message}` }])
            streamingTextRef.current = ''
            setStreamingText('')
            setStatusMessage('')
            setIsLoading(false)
          }
        }
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
      setStatusMessage('')
      setIsLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '760px', margin: '0 auto', background: 'white', boxShadow: '0 0 0 1px #eee' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid #eee', flexShrink: 0 }}>
        <span style={{ fontWeight: '600', fontSize: '16px' }}>✨ Post Writer</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <HeaderButton onClick={() => setDebugMode((d) => !d)} active={debugMode}>
            Debug
          </HeaderButton>
          <HeaderButton onClick={createSession}>New Chat</HeaderButton>
          <HeaderButton onClick={onOpenHistory}>History</HeaderButton>
          <HeaderButton onClick={() => signOut(auth)}>Sign Out</HeaderButton>
        </div>
      </header>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {messages.length === 0 && !isLoading && (
          <p style={{ color: '#bbb', textAlign: 'center', marginTop: '40px', fontSize: '14px' }}>
            Tell me about a place you visited...
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {isLoading && !streamingText && (
          <div style={{ alignSelf: 'flex-start', padding: '10px 14px', borderRadius: '18px 18px 18px 4px', background: '#F0F0F0', color: '#666', fontSize: '14px' }}>
            {statusMessage || '💭 Thinking...'}
          </div>
        )}
        {streamingText && (
          <div style={{ alignSelf: 'flex-start', maxWidth: '78%', padding: '10px 14px', borderRadius: '18px 18px 18px 4px', background: '#F0F0F0', color: '#1a1a1a', fontSize: '14px', lineHeight: '1.5', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {streamingText}
            <span style={{ display: 'inline-block', width: '2px', height: '14px', background: '#999', marginLeft: '2px', verticalAlign: 'middle', animation: 'blink 1s step-end infinite' }} />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Debug panel */}
      {debugMode && <DebugPanel events={debugEvents} />}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{ display: 'flex', gap: '8px', padding: '12px 16px', borderTop: '1px solid #eee', flexShrink: 0 }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Describe your experience..."
          disabled={isLoading}
          style={{ flex: 1, padding: '10px 14px', borderRadius: '20px', border: '1px solid #ddd', fontSize: '14px', outline: 'none' }}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{ padding: '10px 18px', borderRadius: '20px', border: 'none', background: '#007AFF', color: 'white', fontSize: '14px', fontWeight: '500', opacity: isLoading || !input.trim() ? 0.5 : 1 }}
        >
          Send
        </button>
      </form>
    </div>
  )
}

function HeaderButton({ onClick, children, active }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '6px 12px',
        borderRadius: '6px',
        border: '1px solid #ddd',
        background: active ? '#007AFF' : 'white',
        color: active ? 'white' : '#333',
        fontSize: '13px',
      }}
    >
      {children}
    </button>
  )
}
