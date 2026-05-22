export default function MessageBubble({ message, onApply }) {
  const isUser = message.role === 'user'
  return (
    <div style={{ alignSelf: isUser ? 'flex-end' : 'flex-start', maxWidth: '78%', display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div
        style={{
          padding: '10px 14px',
          borderRadius: isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
          background: isUser ? '#007AFF' : '#F0F0F0',
          color: isUser ? 'white' : '#1a1a1a',
          fontSize: '14px',
          lineHeight: '1.5',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {message.content}
      </div>
      {!isUser && message.isDraft && onApply && (
        <button
          onClick={() => onApply(message.content)}
          style={{
            alignSelf: 'flex-start',
            padding: '5px 14px',
            borderRadius: '14px',
            border: '1px solid #007AFF',
            background: 'white',
            color: '#007AFF',
            fontSize: '12px',
            fontWeight: '500',
            cursor: 'pointer',
          }}
        >
          Apply
        </button>
      )}
    </div>
  )
}
