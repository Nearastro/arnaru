import type { VercelRequest, VercelResponse } from '@vercel/node';
import { OpenAIModelsResponse, OpenAIModel } from '../../lib/types';
import { getTimestamp } from '../../lib/arnaru';

// Full list of Arnaru chat models
const ARNARU_MODELS: string[] = [
  'claude-fable-5',
  'claude-haiku-4.5',
  'claude-opus-4.6',
  'claude-opus-4.7',
  'claude-opus-4.8',
  'claude-opus-5-(soon)',
  'claude-sonnet-4',
  'claude-sonnet-4.6',
  'claude-sonnet-5',
  'deepseek-r1',
  'deepseek-v3.1',
  'deepseek-v3.2',
  'deepseek-v3.2-online',
  'deepseek-v3.2-think',
  'deepseek-v4-flash',
  'deepseek-v4-pro',
  'doubao-1.5-pro',
  'doubao-seed-1.8',
  'doubao-v4.1',
  'doubao-v4.2',
  'doubao-v4.5',
  'gemini-2.0-flash',
  'gemini-2.5-flash',
  'gemini-2.5-pro',
  'gemini-3-flash',
  'gemini-3-pro',
  'gemini-3.1-flash',
  'gemini-3.1-pro',
  'gemini-3.5-flash',
  'gemini-3.5-flash-lite',
  'gemini-3.6-flash',
  'gpt-4.1',
  'gpt-4.1-mini',
  'gpt-4o',
  'gpt-5',
  'gpt-5-mini',
  'gpt-5-nano',
  'gpt-5.1',
  'gpt-5.2',
  'gpt-5.4',
  'gpt-5.5',
  'gpt-5.6-luna',
  'gpt-5.6-sol',
  'gpt-o3-mini',
  'grok-3',
  'grok-3-reasoner',
  'grok-4',
  'grok-4-fast',
  'grok-4-reasoning',
  'grok-4.1',
  'grok-4.1-fast',
  'grok-4.1-reasoning',
  'grok-4.2',
  'grok-4.2-reasoning',
  'grok-4.3-pro',
  'grok-4.3-reasoning',
  'grok-4.5',
  'kimi-k3',
  'llama-4',
  'llama-4.1',
  'mistral-small-3.2',
  'mistral-small-creative',
  'qwen-vl-max',
  'qwen3-235b',
  'qwen3-max',
  'skylark-pro',
  'step-3.5-flash',
  'step-3.5-flash-free'
];

export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed. Use GET.' });
  }

  const models: OpenAIModel[] = ARNARU_MODELS.map((id, index) => ({
    id,
    object: 'model',
    created: getTimestamp() - index * 1000, // Stagger timestamps
    owned_by: 'arnaru-ai'
  }));

  const response: OpenAIModelsResponse = {
    object: 'list',
    data: models
  };

  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  return res.status(200).json(response);
}
