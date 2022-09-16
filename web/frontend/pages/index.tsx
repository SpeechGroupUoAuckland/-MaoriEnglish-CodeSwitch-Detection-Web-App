import { useRouter } from 'next/router'
import { useStore } from 'react-redux'
import type { NextPage } from 'next'

const Home: NextPage = () => {
  const router = useRouter()
  const langStates = useStore()
  return (
    <div className="m-auto flex h-screen max-w-sm flex-col items-center justify-center gap-4 p-4">
      <h1 className="text-3xl font-bold capitalize">
        <ruby className="flex flex-col items-center justify-items-center text-4xl font-bold">
          <rp>(</rp>
          <rt>Tīpakohia te Reo</rt>
          <rp>)</rp>
          <h1>Select Language</h1>
        </ruby>
      </h1>
      <div className="flex flex-row justify-center gap-4">
        <button
          className="mt-2 w-full rounded bg-purple-500 px-4 py-8 font-bold text-white shadow hover:bg-purple-400"
          type="button"
          onClick={() => {
            langStates.dispatch({
              type: 'global/setLang',
              payload: 'm',
            })
            router.push('/main')
          }}
        >
          Te Reo Māori
        </button>
        <button
          className="mt-2 w-full rounded bg-purple-500 px-4 py-8 font-bold text-white shadow hover:bg-purple-400"
          type="button"
          onClick={() => {
            langStates.dispatch({
              type: 'global/setLang',
              payload: 'e',
            })
            router.push('/main')
          }}
        >
          English
        </button>
      </div>
    </div>
  )
}

export default Home
