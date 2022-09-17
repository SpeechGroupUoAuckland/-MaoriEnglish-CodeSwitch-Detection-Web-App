import type { NextApiRequest, NextApiResponse } from 'next'

export type Prediction = {
  cleaned: string
  input: string
  labels: string[]
  probability: any[]
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Prediction>
) {
  const errorHandling: Prediction = {
    cleaned: ' ',
    input: ' ',
    labels: [' '],
    probability: [[0, 0]],
  }

  const { model, text } = req.query

  if (typeof model !== 'string' || typeof text !== 'string') {
    res.status(400).json(errorHandling)
  } else {
    await fetch('http://localhost:8500/?model=' + model + '&text=' + text)
      .then((res) => res.json())
      .then((data) => {
        res.status(200).json(data)
      })
      .catch((rejected) => {
        console.log(rejected)
        res.status(400).json(errorHandling)
      })
  }
}
