import { useState } from 'react'
import { GoogleAuthProvider, signInWithEmailAndPassword, signInWithPopup } from 'firebase/auth'
import { auth } from '../firebase'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const loginWithGoogle = async () => {
    setError('')
    setLoading(true)
    try {
      await signInWithPopup(auth, new GoogleAuthProvider())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const loginWithEmail = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await signInWithEmailAndPassword(auth, email, password)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#f5f5f5' }}>
      <div style={{ background: 'white', padding: '40px', borderRadius: '12px', width: '360px', boxShadow: '0 2px 12px rgba(0,0,0,0.08)' }}>
        <h1 style={{ marginBottom: '8px', fontSize: '22px' }}>✨ Post Writer</h1>
        <p style={{ color: '#999', marginBottom: '28px', fontSize: '14px' }}>Sign in to continue</p>

        <button
          onClick={loginWithGoogle}
          disabled={loading}
          style={{ width: '100%', padding: '10px', marginBottom: '20px', borderRadius: '8px', border: '1px solid #ddd', background: 'white', fontSize: '14px', fontWeight: '500' }}
        >
          Sign in with Google
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '20px' }}>
          <div style={{ flex: 1, height: '1px', background: '#eee' }} />
          <span style={{ color: '#bbb', fontSize: '12px' }}>or</span>
          <div style={{ flex: 1, height: '1px', background: '#eee' }} />
        </div>

        <form onSubmit={loginWithEmail} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            required
            style={{ padding: '10px 12px', borderRadius: '8px', border: '1px solid #ddd', fontSize: '14px' }}
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            required
            style={{ padding: '10px 12px', borderRadius: '8px', border: '1px solid #ddd', fontSize: '14px' }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{ padding: '10px', borderRadius: '8px', border: 'none', background: '#007AFF', color: 'white', fontSize: '14px', fontWeight: '500' }}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        {error && (
          <p style={{ marginTop: '16px', color: '#e53e3e', fontSize: '13px' }}>{error}</p>
        )}
      </div>
    </div>
  )
}
