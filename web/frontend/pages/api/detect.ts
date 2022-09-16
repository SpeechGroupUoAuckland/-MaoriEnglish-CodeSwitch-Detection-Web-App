import type { NextApiRequest, NextApiResponse } from 'next'

export type Prediction = {
  cleaned: string
  input: string
  labels: string[]
  probability: any[]
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Prediction[]>
) {
  const { model, text } = req.query

  await fetch('http://localhost:8500/?model=' + model + '&text=' + text)
    .then((res) => res.json())
    .then((data) => {
      res.status(200).json(data)
    })
}
