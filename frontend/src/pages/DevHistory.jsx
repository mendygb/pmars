import { useEffect, useState } from 'react'

const STYLE_COLORS = {
  checkin: '#FF9500',
  recommendation: '#34C759',
  guide: '#007AFF',
  diary: '#AF52DE',
  freeform: '#8E8E93',
}

export default function DevHistory({ user, onBack }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(null)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState(null)
  const [activeTab, setActiveTab] = useState('conversation')

  const getToken = () => user.getIdToken()

  useEffect(() => {
    const load = async () => {
      try {
        const token = await getToken()
        const res = await fetch('/api/sessions', { headers: { Authorization: `Bearer ${token}` } })
        const data = await res.json()
        setSessions(data.sessions)
      } catch (e) {
        console.error(e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const deleteSession = async (e, sessionId, isCompleted) => {
    e.stopPropagation()
    if (!confirm('Delete this session?')) return
    const token = await getToken()
    const url = isCompleted ? `/api/sessions/${sessionId}/completed` : `/api/sessions/${sessionId}`
    await fetch(url, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } })
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
    if (expanded === sessionId) { setExpanded(null); setDetail(null); setDetailError(null) }
  }

  const toggleSession = async (sessionId) => {
    if (expanded === sessionId) {
      setExpanded(null)
      setDetail(null)
      setDetailError(null)
      return
    }
    setExpanded(sessionId)
    setDetail(null)
    setDetailError(null)
    setDetailLoading(true)
    setActiveTab('conversation')
    try {
      const token = await getToken()
      const res = await fetch(`/api/sessions/${sessionId}`, { headers: { Authorization: `Bearer ${token}` } })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        setDetailError(`Error ${res.status}: ${err.detail ?? res.statusText}`)
        return
      }
      setDetail(await res.json())
    } catch (e) {
      console.error(e)
      setDetailError(e.message)
    } finally {
      setDetailLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '820px', margin: '0 auto', height: '100vh', display: 'flex', flexDirection: 'column', background: 'white', boxShadow: '0 0 0 1px #eee' }}>
      {/* Header */}
      <header style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderBottom: '1px solid #eee', flexShrink: 0 }}>
        <button onClick={onBack} style={{ padding: '6px 12px', borderRadius: '6px', border: '1px solid #ddd', background: 'white', fontSize: '13px', cursor: 'pointer' }}>
          ← Back
        </button>
        <span style={{ fontWeight: '600', fontSize: '16px' }}>Dev History</span>
        <span style={{ color: '#999', fontSize: '13px' }}>({sessions.length} sessions)</span>
      </header>

      {/* Session list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
        {loading && <p style={{ color: '#999', padding: '16px' }}>Loading...</p>}
        {!loading && sessions.length === 0 && (
          <p style={{ color: '#999', padding: '16px' }}>No sessions yet. Start a chat first.</p>
        )}

        {sessions.map((s) => (
          <div key={s.session_id} style={{ marginBottom: '8px', borderRadius: '10px', border: '1px solid #eee', overflow: 'hidden' }}>
            {/* Session row */}
            <button
              onClick={() => toggleSession(s.session_id)}
              style={{ width: '100%', display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 14px', background: expanded === s.session_id ? '#f8f8f8' : 'white', border: 'none', cursor: 'pointer', textAlign: 'left' }}
            >
              <StyleBadge style={s.style} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '14px', color: '#1a1a1a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {s.preview}
                </div>
                <div style={{ fontSize: '12px', color: '#999', marginTop: '2px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {s.turn_count} turn{s.turn_count !== 1 ? 's' : ''} · {formatTime(s.updated_at)}
                  {s.completed && (
                    <span style={{ padding: '1px 6px', borderRadius: '10px', background: '#e8f5e9', color: '#2e7d32', fontSize: '11px', fontWeight: '600' }}>
                      Applied ✓
                    </span>
                  )}
                </div>
              </div>
              <span style={{ color: '#ccc', fontSize: '12px' }}>{expanded === s.session_id ? '▲' : '▼'}</span>
              <button
                onClick={(e) => deleteSession(e, s.session_id, s.completed)}
                style={{ padding: '3px 8px', borderRadius: '5px', border: '1px solid #ffcccc', background: '#fff5f5', color: '#e53e3e', fontSize: '12px', flexShrink: 0 }}
              >
                Delete
              </button>
            </button>

            {/* Expanded detail */}
            {expanded === s.session_id && (
              <div style={{ borderTop: '1px solid #eee' }}>
                {detailLoading && <p style={{ color: '#999', padding: '16px', fontSize: '13px' }}>Loading detail...</p>}
                {detailError && <p style={{ color: '#e53e3e', padding: '16px', fontSize: '13px' }}>{detailError}</p>}
                {detail && (
                  <>
                    {/* Tabs */}
                    <div style={{ display: 'flex', borderBottom: '1px solid #eee', background: '#fafafa' }}>
                      {['conversation', 'debug'].map((tab) => (
                        <button
                          key={tab}
                          onClick={() => setActiveTab(tab)}
                          style={{ padding: '8px 16px', border: 'none', background: 'none', fontSize: '13px', fontWeight: activeTab === tab ? '600' : '400', color: activeTab === tab ? '#007AFF' : '#666', borderBottom: activeTab === tab ? '2px solid #007AFF' : '2px solid transparent', cursor: 'pointer' }}
                        >
                          {tab.charAt(0).toUpperCase() + tab.slice(1)}
                        </button>
                      ))}
                    </div>

                    {activeTab === 'conversation' && (
                      <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {detail.history.length === 0 && (
                          <p style={{ color: '#999', fontSize: '13px' }}>No conversation turns yet.</p>
                        )}
                        {detail.history.map((turn, i) => (
                          <div
                            key={i}
                            style={{
                              alignSelf: turn.role === 'user' ? 'flex-end' : 'flex-start',
                              maxWidth: '80%',
                              padding: '8px 12px',
                              borderRadius: turn.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                              background: turn.role === 'user' ? '#007AFF' : '#F0F0F0',
                              color: turn.role === 'user' ? 'white' : '#1a1a1a',
                              fontSize: '13px',
                              lineHeight: '1.5',
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                            }}
                          >
                            {turn.content}
                          </div>
                        ))}
                      </div>
                    )}

                    {activeTab === 'debug' && (
                      <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {/* Pipeline state */}
                        <Section title="Pipeline State">
                          <DebugRow label="Style" value={<StyleBadge style={detail.debug.style} inline />} />
                          <DebugRow label="Last route" value={detail.debug.next_node || '—'} />
                          <DebugRow label="Needs clarification" value={String(detail.debug.needs_clarification)} />
                          <DebugRow label="Web search used" value={String(detail.debug.web_search_used)} />
                        </Section>

                        {/* Draft vs Final */}
                        {detail.debug.draft_content && (
                          <Section title="Draft (before Critic)">
                            <pre style={{ margin: 0, fontFamily: 'inherit', fontSize: '13px', whiteSpace: 'pre-wrap', color: '#555' }}>
                              {detail.debug.draft_content}
                            </pre>
                          </Section>
                        )}
                        {detail.debug.final_post && (
                          <Section title="Final Post (after Critic)">
                            <pre style={{ margin: 0, fontFamily: 'inherit', fontSize: '13px', whiteSpace: 'pre-wrap', color: '#1a1a1a' }}>
                              {detail.debug.final_post}
                            </pre>
                          </Section>
                        )}

                        {/* Latency */}
                        {detail.debug.timings && Object.keys(detail.debug.timings).length > 0 && (
                          <Section title="Latency">
                            {detail.debug.timings.total_ms !== undefined && (
                              <DebugRow label="Total" value={`${detail.debug.timings.total_ms} ms`} />
                            )}
                            {['director', 'research', 'copywriter', 'safety_check', 'critic'].map((node) =>
                              detail.debug.timings[node] !== undefined ? (
                                <DebugRow key={node} label={node} value={`${detail.debug.timings[node]} ms`} />
                              ) : null
                            )}
                          </Section>
                        )}

                        {/* RAG metrics */}
                        {Object.keys(detail.debug.rag_metrics).length > 0 && (
                          <Section title="RAG Metrics">
                            {Object.entries(detail.debug.rag_metrics).map(([k, v]) => (
                              <DebugRow key={k} label={k} value={typeof v === 'number' ? `${v} ms` : String(v)} />
                            ))}
                          </Section>
                        )}

                        {/* Raw state dump */}
                        <Section title="Raw Debug JSON">
                          <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '11px', color: '#555', whiteSpace: 'pre-wrap', overflowX: 'auto' }}>
                            {JSON.stringify(detail.debug, null, 2)}
                          </pre>
                        </Section>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function StyleBadge({ style, inline }) {
  const color = STYLE_COLORS[style] || STYLE_COLORS.freeform
  return (
    <span style={{
      padding: inline ? '1px 7px' : '3px 8px',
      borderRadius: '20px',
      background: color + '22',
      color,
      fontSize: '11px',
      fontWeight: '600',
      whiteSpace: 'nowrap',
      flexShrink: 0,
    }}>
      {style || 'new'}
    </span>
  )
}

function Section({ title, children }) {
  return (
    <div style={{ border: '1px solid #eee', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ padding: '6px 12px', background: '#f8f8f8', fontSize: '11px', fontWeight: '600', color: '#666', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {title}
      </div>
      <div style={{ padding: '10px 12px' }}>
        {children}
      </div>
    </div>
  )
}

function DebugRow({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '3px 0', fontSize: '13px' }}>
      <span style={{ color: '#666' }}>{label}</span>
      <span style={{ color: '#1a1a1a', fontWeight: '500' }}>{value}</span>
    </div>
  )
}

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
