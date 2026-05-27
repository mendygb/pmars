import { useEffect, useRef, useState } from 'react'
import MessageBubble from '../components/MessageBubble'
import DebugPanel from '../components/DebugPanel'

export default function Chat({ user, location, initialContent, seedDraft, debugMode, onDebugToggle, onApply, onBack }) {
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [streamingText, setStreamingText] = useState('')
  const [debugEvents, setDebugEvents] = useState([])
  const messagesEndRef = useRef(null)
  const streamingTextRef = useRef('')
  const sessionIdRef = useRef(null)
  const autoSentRef = useRef(false)
  const initialSessionRef = useRef(false)
  const inputRef = useRef(null)
  const abortControllerRef = useRef(null)

  const getToken = () => user.getIdToken()

  const createSession = async (isInitial = false) => {
    if (isInitial) {
      // StrictMode double-invokes mount effects — guard so only the first call proceeds
      if (initialSessionRef.current) return
      initialSessionRef.current = true
    } else {
      // "New Chat" button: reset everything synchronously before the async fetch so
      // there's no race between setMessages([]) and the bubble added by sendMessage
      autoSentRef.current = false
      setMessages([])
      setDebugEvents([])
      setInput('')
      streamingTextRef.current = ''
      setStreamingText('')
      setIsLoading(false)
      setStatusMessage('')
    }

    try {
      const token = await getToken()
      const body = seedDraft ? JSON.stringify({ draft_content: seedDraft }) : null
      const res = await fetch('/api/sessions/new', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          ...(seedDraft ? { 'Content-Type': 'application/json' } : {}),
        },
        body,
      })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data = await res.json()
      sessionIdRef.current = data.session_id
      setSessionId(data.session_id)
      // NOTE: do NOT call setMessages([]) here — on initial mount messages already
      // start as [], and calling it after the fetch would race with sendMessage's bubble

      if (initialContent && !autoSentRef.current) {
        autoSentRef.current = true
        sendMessage(initialContent, data.session_id, true)
      }
    } catch (e) {
      console.error('Failed to create session:', e)
    }
  }

  useEffect(() => { createSession(true) }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Auto-resize textarea as content grows
  const resizeInput = () => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  const handleApply = async (finalContent) => {
    if (!sessionId) return
    try {
      const token = await getToken()
      await fetch(`/api/sessions/${sessionId}/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ final_content: finalContent }),
      })
    } catch (e) {
      console.error('Failed to complete session:', e)
    }
    onApply(finalContent)
  }

  const handleCancel = async () => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
    setStatusMessage('')
    setStreamingText('')
    streamingTextRef.current = ''
    if (sessionIdRef.current) {
      try {
        const token = await getToken()
        await fetch(`/api/sessions/${sessionIdRef.current}/cancel`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        })
      } catch (_) {}
    }
  }

  const sendMessage = async (userMessage, currentSessionId, isFirstMessage) => {
    if (!userMessage || !currentSessionId) return

    streamingTextRef.current = ''
    setStreamingText('')

    // Inject location prefix on the first message so Director/Research has place context
    const messageToSend = isFirstMessage && location
      ? `[Location: ${location}] ${userMessage}`
      : userMessage

    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)

    try {
      const token = await getToken()
      abortControllerRef.current = new AbortController()
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ user_input: messageToSend, display_input: userMessage, session_id: currentSessionId, debug: debugMode }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let event
          try { event = JSON.parse(line.slice(6)) } catch { continue }

          if (event.type === 'status') {
            setStatusMessage(event.message)
          } else if (event.type === 'token') {
            streamingTextRef.current += event.content
            setStreamingText(streamingTextRef.current)
          } else if (event.type === 'message_end') {
            const text = streamingTextRef.current
            streamingTextRef.current = ''
            setStreamingText('')
            if (text) setMessages((prev) => [...prev, { role: 'assistant', content: text, isDraft: true }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'message') {
            streamingTextRef.current = ''
            setStreamingText('')
            setMessages((prev) => [...prev, { role: 'assistant', content: event.content, isDraft: !event.is_error }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'debug') {
            setDebugEvents((prev) => [...prev, { node: event.node, payload: event.payload, ts: Date.now() }])
          } else if (event.type === 'clarification') {
            setMessages((prev) => [...prev, { role: 'assistant', content: event.payload.question, isDraft: false }])
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'done') {
            setStatusMessage('')
            setIsLoading(false)
          } else if (event.type === 'cancelled') {
            streamingTextRef.current = ''
            setStreamingText('')
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
      if (err.name === 'AbortError') return
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${err.message}` }])
      setStatusMessage('')
      setIsLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e?.preventDefault()
    const userMessage = input.trim()
    if (!userMessage || isLoading || !sessionId) return
    const isFirstMessage = messages.length === 0
    setInput('')
    // Reset textarea height after clearing
    if (inputRef.current) inputRef.current.style.height = 'auto'
    sendMessage(userMessage, sessionId, isFirstMessage)
  }

  const inputPlaceholder = 'Describe your experience...'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '760px', margin: '0 auto', background: 'white', boxShadow: '0 0 0 1px #eee' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid #eee', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button onClick={onBack} style={headerBtnStyle}>← Back</button>
          <span style={{ fontWeight: '600', fontSize: '16px' }}>✨ Post Writer</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <HeaderButton onClick={onDebugToggle} active={debugMode}>Debug</HeaderButton>
          <HeaderButton onClick={createSession}>New Chat</HeaderButton>
        </div>
      </header>

      {/* Image placeholder with location */}
      <div style={{ flexShrink: 0, height: '120px', background: '#f5f5f5', borderBottom: '1px solid #eee', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '6px', color: '#bbb' }}>
        <span style={{ fontSize: '24px' }}>📷</span>
        {location
          ? <span style={{ fontSize: '13px', color: '#888' }}>📍 {location}</span>
          : <span style={{ fontSize: '13px' }}>Photo</span>
        }
      </div>

      {/* Messages */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {messages.length === 0 && !isLoading && (
          <p style={{ color: '#bbb', textAlign: 'center', marginTop: '40px', fontSize: '14px' }}>
            Tell me about your experience...
          </p>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} onApply={handleApply} />
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

      {debugMode && <DebugPanel events={debugEvents} />}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', padding: '12px 16px', borderTop: '1px solid #eee', flexShrink: 0 }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => { setInput(e.target.value); resizeInput() }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit()
            }
          }}
          placeholder={inputPlaceholder}
          disabled={isLoading}
          rows={1}
          style={{ flex: 1, padding: '10px 14px', borderRadius: '20px', border: '1px solid #ddd', fontSize: '14px', outline: 'none', resize: 'none', lineHeight: '1.5', overflow: 'hidden', fontFamily: 'inherit' }}
        />
        {isLoading ? (
          <button
            type="button"
            onClick={handleCancel}
            style={{ padding: '10px 18px', borderRadius: '20px', border: 'none', background: '#FF3B30', color: 'white', fontSize: '14px', fontWeight: '500', cursor: 'pointer', flexShrink: 0 }}
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            style={{ padding: '10px 18px', borderRadius: '20px', border: 'none', background: '#007AFF', color: 'white', fontSize: '14px', fontWeight: '500', cursor: 'pointer', opacity: !input.trim() ? 0.5 : 1, flexShrink: 0 }}
          >
            Send
          </button>
        )}
      </form>
    </div>
  )
}

const headerBtnStyle = {
  padding: '6px 12px',
  borderRadius: '6px',
  border: '1px solid #ddd',
  background: 'white',
  color: '#333',
  fontSize: '13px',
  cursor: 'pointer',
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
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}
