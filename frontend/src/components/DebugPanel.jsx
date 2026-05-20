import { useEffect, useRef } from 'react'

const NODE_COLORS = {
  director: '#FFD700',
  research: '#00BFFF',
  copywriter: '#7CFC00',
  critic: '#FF69B4',
}

export default function DebugPanel({ events }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  return (
    <div style={{ maxHeight: '180px', overflowY: 'auto', background: '#111', borderTop: '1px solid #333', padding: '8px 12px' }}>
      {events.length === 0 ? (
        <span style={{ color: '#555', fontFamily: 'monospace', fontSize: '12px' }}>Debug panel — events will appear here.</span>
      ) : (
        events.map((ev, i) => (
          <div key={i} style={{ fontFamily: 'monospace', fontSize: '12px', marginBottom: '3px', lineHeight: '1.4' }}>
            <span style={{ color: NODE_COLORS[ev.node] ?? '#ccc', fontWeight: 'bold' }}>[{ev.node}]</span>{' '}
            <span style={{ color: '#ccc' }}>{JSON.stringify(ev.payload)}</span>
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  )
}
