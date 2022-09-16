import type { NextApiRequest, NextApiResponse } from 'next'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<string[]>
) {
  const modelList = await fetch('http://localhost:8500/getModel').then((res) =>
    res.json()
  )

  const result: string[] = []
  modelList.models.forEach((model: string) => {
    result.push(model)
  })

  res.status(200).json(result)
}
