/* eslint-disable @next/next/no-img-element */
import '../styles/globals.css'
import App, { AppContext } from 'next/app'
import type { AppProps } from 'next/app'
import { Provider } from 'react-redux'
import langStates from '../util/langStates'
import Head from 'next/head'

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <Provider store={langStates}>
      <Head>
        <title>M/E CW Detection</title>
        {/* <link rel="icon" href="https://aotearoavoices.nz/favicon.ico" /> */}
      </Head>
      <header className="fixed top-0 left-0 right-0 flex flex-row items-center justify-between bg-gray-200 p-4 opacity-100">
        <a
          href="https://www.auckland.ac.nz/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img
            src="https://blogs.auckland.ac.nz/files/2016/07/UOA-NT-HC-RGB-1dr6b6b.png"
            className="h-12"
            alt="University of Auckland"
          />
        </a>
        <ruby className="flex flex-col items-center justify-items-center text-2xl font-bold">
          <rp>(</rp>
          <rt>Utauta Waehere Whakawhiti Waehere Ingarihi Māori</rt>
          <rp>)</rp>
          <h1>English Māori Code-switching Point Detection Tool</h1>
        </ruby>
        <a
          href="https://speechresearch.auckland.ac.nz/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <img
            src="https://avatars.githubusercontent.com/u/100390597"
            className="h-12"
            alt="Speech Research Group @ University of Auckland"
          />
        </a>
      </header>
      <Component {...pageProps} />
      <footer className="fixed bottom-0 left-0 right-0 flex h-8 w-full items-center justify-center border-t bg-gray-200 opacity-100">
        <label>
          Copyright © 2022 {}
          <a
            href="https://speechresearch.auckland.ac.nz/"
            target="_blank"
            className="font-bold text-blue-600"
            rel="noopener noreferrer"
          >
            Speech Research Group @ UoA
          </a>
          . All rights reserved.
        </label>
      </footer>
      <button
        className="h-15 w-25 sticky bottom-20 left-2 rounded bg-lime-600 px-1 py-3 font-bold text-white shadow hover:bg-lime-700"
        type="button"
        onClick={() =>
          window.open('https://forms.gle/K73u9x394zzneYsB6', '_blank')
        }
      >
        Feedback
      </button>
    </Provider>
  )
}

MyApp.getInitialProps = async (appContext: AppContext) => {
  const appProps = await App.getInitialProps(appContext)
  if (appContext.ctx.res?.statusCode === 404) {
    appContext.ctx.res.writeHead(302, { Location: '/' })
    appContext.ctx.res.end()
    return
  } else {
    return { ...appProps }
  }
}

export default MyApp
