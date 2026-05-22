import { useEffect, useState } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from './firebase'
import Login from './pages/Login'
import EditingPage from './pages/EditingPage'
import Chat from './pages/Chat'
import DevHistory from './pages/DevHistory'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState('editing')
  const [editingState, setEditingState] = useState({ location: '', title: '', content: '' })
  const [hasApplied, setHasApplied] = useState(false)
  const [debugMode, setDebugMode] = useState(false)

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u)
      setLoading(false)
      if (!u) setPage('editing')
    })
    return unsubscribe
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: '#999' }}>
        Loading...
      </div>
    )
  }

  if (!user) return <Login />

  if (page === 'editing') {
    return (
      <EditingPage
        user={user}
        editingState={editingState}
        onEditChange={setEditingState}
        onOpenChat={() => setPage('chat')}
        onOpenHistory={() => setPage('history')}
        onNewPost={() => {
          setEditingState({ location: '', title: '', content: '' })
          setHasApplied(false)
        }}
      />
    )
  }

  if (page === 'chat') {
    return (
      <Chat
        user={user}
        location={editingState.location}
        initialContent={editingState.content}
        seedDraft={hasApplied ? editingState.content : ''}
        debugMode={debugMode}
        onDebugToggle={() => setDebugMode((d) => !d)}
        onApply={(finalContent) => {
          setHasApplied(true)
          setEditingState((prev) => ({ ...prev, content: finalContent }))
          setPage('editing')
        }}
        onBack={() => setPage('editing')}
      />
    )
  }

  return <DevHistory user={user} onBack={() => setPage('editing')} />
}
