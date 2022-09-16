import { createSlice, configureStore } from '@reduxjs/toolkit'

export interface GlobalState {
  lang?: string
  eList?: string[]
  mList?: string[]
}

const initialState: GlobalState = {
  lang: undefined,
  eList: [
    'This tool detects the code-switching point of English and Māori.',
    'English and Māori only. The maximum number of characters is 2500. Non Ascii and non Māori characters will be ignored.',
    'Start typing here...',
    'Select a model:',
    'Detect',
    'Result',
    'To display probability of language detection',
    'Display the result in verbose mode. A probability list will be displayed for each vocabulary word on the top. The first entry is the probability of the word being Māori, the second is the probability of the word being English, and 1 - sum(probability list) is the probability of being ambiguous.If a word is more likely to be Māori, it will be rendered dark yellow, otherwise, it will be rendered grey. It will be rendered in a cyan colour if the probability is equal.',
    'For more information of the definition of code switch detection and the models used, please visit: ',
  ],
  mList: [
    'Ka kitea e tenei taputapu te waahi whakawhiti-waehere o te reo Ingarihi me te reo Maori.',
    'Ko te reo Ingarihi me te reo Maori anake. Ko te nui rawa o nga tohu ko te 2500. Ko nga tohu ehara i te Ascii me te reo Maori ka warewarehia.',
    'Tīmata pato ki konei...',
    'Tīpakohia he tauira:',
    'Rapu',
    'Hua',
    'Hei whakaatu i te tupono ka kitea te reo',
    'Whakaatuhia te hua ki te aratau kupu. Ka whakaatuhia he rarangi tūponotanga mo ia kupu kupu kei runga. Ko te urunga tuatahi ko te tupono o te kupu he Maori, ko te tuarua ko te tupono o te kupu he pakeha, me te 1 - sum( rārangi tūponotanga) ko te tūponotanga he rangirua. Ki te mea he reo Maori te kupu, ka puta he kowhai pouri, ki te kore, ka puta ke ki te hina. Ka puta ki te tae cyan mena he rite te tupono.',
    'Mo etahi atu korero mo te whakamaramatanga o te kitenga whakawhiti waehere me nga tauira i whakamahia, tirohia koa: ',
  ],
}

const globalSlice = createSlice({
  name: 'global',
  initialState,
  reducers: {
    setLang: (state, action) => {
      state.lang = action.payload
    },
  },
})

const langStates = configureStore({
  reducer: {
    global: globalSlice.reducer,
  },
})

// langStates.subscribe(() => console.log(langStates.getState()));

export default langStates
