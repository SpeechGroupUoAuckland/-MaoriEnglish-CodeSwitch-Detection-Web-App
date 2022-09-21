import { useRouter } from 'next/router'
import { useEffect, useState } from 'react'
import { useStore } from 'react-redux'
import {
  TextField,
  MenuItem,
  Switch,
  Snackbar,
  CircularProgress,
} from '@mui/material'

import { BiInfoCircle } from 'react-icons/bi'

import { Prediction } from './api/detect'

export default function StartPage() {
  const router = useRouter()
  const langStates = useStore()
  const lang = langStates.getState() as any
  const [uiSentenceList, setUiSentenceList] = useState<string[]>()

  const [langState, setLangState] = useState<boolean>(lang?.global.lang === 'e')

  const [inputText, setInputText] = useState<string>('')
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [isLoading, setLoadingState] = useState(false)
  const [prediction, setPrediction] = useState<Prediction>()
  const [verbose, setVerbose] = useState<boolean>(false)
  const [showVerboseInfo, setShowVerboseInfo] = useState<boolean>(false)

  useEffect(() => {
    if (lang?.global.lang === undefined) {
      router.push('/')
      return
    }

    if (lang?.global.lang === 'e') {
      setUiSentenceList(lang?.global.eList)
    } else if (lang?.global.lang === 'm') {
      setUiSentenceList(lang?.global.mList)
    }

    fetch('/api/getModel')
      .then((res) => res.json())
      .then((data) => {
        setModels(data)
        setSelectedModel(data[3])
      })
      .catch((rejected) => {
        console.log(rejected)
      })
  }, [lang, router, langState])

  function renderPrediction(result: Prediction | undefined, verbose: boolean) {
    const r = result ? result : undefined

    if (r) {
      const threshold = 0.95

      const tl = r.cleaned.split(' ')
      // const l = r.labels;
      const pp = r.probability
      const p: any[] = []

      // check if length of pp and tl is 0
      if (
        !tl ||
        !pp ||
        pp.length === 0 ||
        tl.length === 0 ||
        pp.length !== tl.length
      ) {
        return <div></div>
      } else {
        const e: any[] = []

        pp.forEach((item) => {
          const tmpList: number[] = []
          item.forEach((child: string) => {
            tmpList.push(parseFloat(child))
          })
          p.push(tmpList)
        })

        if (verbose) {
          tl.forEach((t, i) => {
            if (p[i][0] >= threshold) {
              e[i] = (
                <ruby className="flex flex-col place-items-center bg-[#ffff00]">
                  <rp>(</rp>
                  <rt>[{pp[i].toString()}]</rt>
                  <rp>)</rp>
                  {t}
                </ruby>
              )
            } else if (p[i][1] >= threshold) {
              e[i] = (
                <ruby className="flex flex-col place-items-center">
                  <rp>(</rp>
                  <rt>[{pp[i].toString()}]</rt>
                  <rp>)</rp>
                  {t}
                </ruby>
              )
            } else if (p[i][0] > p[i][1]) {
              e[i] = (
                <ruby className="flex flex-col place-items-center bg-orange-300">
                  <rp>(</rp>
                  <rt>[{pp[i].toString()}]</rt>
                  <rp>)</rp>
                  {t}
                </ruby>
              )
            } else if (p[i][1] > p[i][0]) {
              e[i] = (
                <ruby className="flex flex-col place-items-center bg-neutral-300">
                  <rp>(</rp>
                  <rt>[{pp[i].toString()}]</rt>
                  <rp>)</rp>
                  {t}
                </ruby>
              )
            } else {
              e[i] = (
                <ruby className="flex flex-col place-items-center bg-cyan-300">
                  <rp>(</rp>
                  <rt>[{pp[i].toString()}]</rt>
                  <rp>)</rp>
                  {t}
                </ruby>
              )
            }
          })
        } else {
          tl.forEach((t, i) => {
            if (p[i][0] >= threshold) {
              e[i] = (
                <div className="flex flex-col place-items-center bg-[#ffff00]">
                  {t}
                </div>
              )
            } else if (p[i][1] >= threshold) {
              e[i] = <div className="flex flex-col place-items-center">{t}</div>
            } else if (p[i][0] > p[i][1]) {
              e[i] = (
                <div className="flex flex-col place-items-center bg-orange-300">
                  {t}
                </div>
              )
            } else if (p[i][1] > p[i][0]) {
              e[i] = (
                <div className="flex flex-col place-items-center bg-neutral-300">
                  {t}
                </div>
              )
            } else {
              e[i] = (
                <div className="flex flex-col place-items-center bg-cyan-300">
                  {t}
                </div>
              )
            }
          })
        }

        return (
          <div className="flex flex-row flex-wrap gap-1">
            {e?.map((item: any, i: number) => (
              <div key={i}>{item}</div>
            ))}
          </div>
        )
      }
    } else {
      return <div></div>
    }
  }

  return (
    <div className="h-full w-full">
      <button
        className={
          langState
            ? 'sticky top-60 left-2 h-20 w-20 rounded bg-neutral-600 px-1 py-3 font-bold text-white shadow hover:bg-neutral-500'
            : 'sticky top-60 left-2 h-20 w-20 rounded bg-amber-600 px-1 py-3 font-bold text-white shadow hover:bg-amber-500'
        }
        type="button"
        onClick={() => {
          {
            langState
              ? langStates.dispatch({
                  type: 'global/setLang',
                  payload: 'm',
                })
              : langStates.dispatch({
                  type: 'global/setLang',
                  payload: 'e',
                })
          }
          setLangState(!langState)
        }}
      >
        {langState ? 'English' : 'Te Reo Māori'}
      </button>

      <div className="m-auto flex w-10/12 flex-col items-start justify-start gap-8 pt-24 pb-12">
        <TextField
          id="filled-textarea"
          multiline
          fullWidth
          label={uiSentenceList ? uiSentenceList[0] : ''}
          helperText={uiSentenceList ? uiSentenceList[1] : ''}
          placeholder={uiSentenceList ? uiSentenceList[2] : ''}
          rows={8}
          variant="filled"
          onChange={(e) => {
            setInputText(e.target.value)
          }}
        />
        <div className="mt-8 text-xl font-bold">
          {uiSentenceList ? uiSentenceList[3] : ''}
        </div>
        <div className="flex w-full flex-row items-center justify-between gap-4">
          <TextField
            sx={{ width: '25ch' }}
            id="outlined-select-currency"
            select
            label={uiSentenceList ? uiSentenceList[3].split(' ')[0] : ''}
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value)
            }}
          >
            {models.map((model) => (
              <MenuItem key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </TextField>
          {isLoading ? (
            <button
              className="w-36 rounded bg-purple-100 px-4 py-4 font-bold text-white"
              type="button"
              disabled
            >
              Detect
            </button>
          ) : (
            <button
              className="w-36 rounded bg-purple-500 px-4 py-4 font-bold text-white shadow hover:bg-purple-400"
              type="button"
              onClick={() => {
                setLoadingState(true)
                fetch(
                  '/api/detect' +
                    '?model=' +
                    selectedModel +
                    '&text=' +
                    inputText
                )
                  .then((res) => res.json())
                  .then((data) => {
                    setPrediction(data)
                  })
                  .catch((rejected) => {
                    console.log(rejected)
                  })

                setTimeout(() => {
                  setLoadingState(false)
                }, 600)
                // setLoadingState(false)
              }}
            >
              {uiSentenceList ? uiSentenceList[4] : ''}
            </button>
          )}
        </div>
        <div className="mt-8 flex w-full flex-row items-center justify-between gap-4">
          <div className="text-xl font-bold">
            {uiSentenceList ? uiSentenceList[5] : ''}
          </div>
          <div className="flex flex-row items-center justify-center gap-4">
            <Switch
              checked={verbose}
              onChange={() => setVerbose(!verbose)}
              name="verbose"
              color="primary"
            />
            {uiSentenceList ? uiSentenceList[6] : ''}
            <div className="items-center justify-center">
              <BiInfoCircle
                className="h-6 w-6 cursor-pointer text-gray-500"
                onClick={() => {
                  setShowVerboseInfo(!showVerboseInfo)
                }}
              />
            </div>
          </div>
        </div>
        {isLoading ? (
          <div className="flex h-96 w-full flex-row items-center justify-center rounded bg-gray-100 shadow">
            <CircularProgress />
          </div>
        ) : (
          <div className="h-96 w-full rounded bg-gray-100 p-2 shadow">
            {renderPrediction(prediction ? prediction : undefined, verbose)}
          </div>
        )}
        <div className="mt-8 text-sm">
          {uiSentenceList ? uiSentenceList[8] : ''}
          <a
            href="https://openreview.net/forum?id=rAxl_GibSWq"
            className="text-sky-500"
            target="_blank"
            rel="noopener noreferrer"
          >
            https://openreview.net/forum?id=rAxl_GibSWq
          </a>
        </div>
        {showVerboseInfo ? (
          <Snackbar
            anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            open={showVerboseInfo}
            autoHideDuration={6000}
            ContentProps={{
              sx: {
                background: '#2288cc',
              },
            }}
            onClose={() => {
              setShowVerboseInfo(false)
            }}
            message={uiSentenceList ? uiSentenceList[7] : ''}
          />
        ) : (
          <div></div>
        )}
      </div>
    </div>
  )
}
