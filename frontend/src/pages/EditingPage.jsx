import { useState } from 'react'
import { signOut } from 'firebase/auth'
import { auth } from '../firebase'

export default function EditingPage({ user, editingState, onEditChange, onOpenChat, onOpenHistory, onNewPost }) {
  const { location, title, content } = editingState
  const [prevTitle, setPrevTitle] = useState(null)
  const [titleLoading, setTitleLoading] = useState(false)

  const update = (field) => (e) => {
    if (field === 'title') setPrevTitle(null)  // manual edit clears undo
    onEditChange((prev) => ({ ...prev, [field]: e.target.value }))
  }

  const generateTitle = async () => {
    if (!content && !title) return
    setTitleLoading(true)
    try {
      const token = await user.getIdToken()
      const res = await fetch('/api/generate-title', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ content: content.slice(0, 600), title }),
      })
      if (!res.ok) throw new Error('Failed')
      const data = await res.json()
      setPrevTitle(title)
      onEditChange((prev) => ({ ...prev, title: data.title }))
    } catch (e) {
      console.error('Title generation failed:', e)
    } finally {
      setTitleLoading(false)
    }
  }

  const undoTitle = () => {
    onEditChange((prev) => ({ ...prev, title: prevTitle }))
    setPrevTitle(null)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '760px', margin: '0 auto', background: 'white', boxShadow: '0 0 0 1px #eee' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid #eee', flexShrink: 0 }}>
        <span style={{ fontWeight: '600', fontSize: '16px' }}>New Post</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          {(location || title || content) && (
            <button onClick={onNewPost} style={btnStyle}>Clear</button>
          )}
          <button onClick={onOpenHistory} style={btnStyle}>History</button>
          <button onClick={() => signOut(auth)} style={btnStyle}>Sign Out</button>
        </div>
      </header>

      {/* Form */}
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '20px 16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Mock image placeholder */}
        <div style={{ width: '100%', height: '180px', background: '#f5f5f5', borderRadius: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#bbb', border: '2px dashed #e0e0e0', flexShrink: 0 }}>
          <span style={{ fontSize: '32px' }}>📷</span>
          <span style={{ fontSize: '13px' }}>Photo</span>
        </div>

        {/* Location */}
        <input
          value={location}
          onChange={update('location')}
          placeholder="Where did you go?"
          style={inputStyle}
        />

        {/* Title row with generate + undo */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <input
            value={title}
            onChange={update('title')}
            placeholder="Add a title"
            style={{ ...inputStyle, flex: 1, width: 'auto' }}
          />
          <button
            onClick={generateTitle}
            disabled={titleLoading || (!content && !title)}
            title="Generate title"
            style={{
              ...iconBtnStyle,
              opacity: titleLoading || (!content && !title) ? 0.4 : 1,
            }}
          >
            {titleLoading ? '…' : '✨'}
          </button>
          {prevTitle !== null && (
            <button onClick={undoTitle} style={btnStyle} title="Restore previous title">
              Undo
            </button>
          )}
        </div>

        {/* Content */}
        <textarea
          value={content}
          onChange={update('content')}
          placeholder="Share your experience..."
          rows={6}
          style={{ ...inputStyle, resize: 'vertical', lineHeight: '1.5' }}
        />

        {/* AI assist button */}
        <button
          onClick={onOpenChat}
          style={{ padding: '12px', borderRadius: '10px', border: 'none', background: '#007AFF', color: 'white', fontSize: '15px', fontWeight: '600', cursor: 'pointer' }}
        >
          Write with AI
        </button>
      </div>
    </div>
  )
}

const inputStyle = {
  padding: '12px 14px',
  borderRadius: '10px',
  border: '1px solid #ddd',
  fontSize: '14px',
  outline: 'none',
  width: '100%',
  boxSizing: 'border-box',
}

const btnStyle = {
  padding: '6px 12px',
  borderRadius: '6px',
  border: '1px solid #ddd',
  background: 'white',
  color: '#333',
  fontSize: '13px',
  cursor: 'pointer',
  flexShrink: 0,
}

const iconBtnStyle = {
  padding: '8px 10px',
  borderRadius: '8px',
  border: '1px solid #ddd',
  background: 'white',
  fontSize: '16px',
  cursor: 'pointer',
  flexShrink: 0,
  lineHeight: 1,
}
