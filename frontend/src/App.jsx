import { useEffect, useState } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from './firebase'
import Login from './pages/Login'
import Chat from './pages/Chat'
import DevHistory from './pages/DevHistory'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState('chat')

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u)
      setLoading(false)
      if (!u) setPage('chat')
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
  return (
    <>
      <div style={{ display: page === 'chat' ? 'block' : 'none' }}>
        <Chat user={user} onOpenHistory={() => setPage('history')} />
      </div>
      {page === 'history' && <DevHistory user={user} onBack={() => setPage('chat')} />}
    </>
  )
}
