export interface OpenAITextPart {
  type: 'text';
  text: string;
}

export interface OpenAIImageUrlPart {
  type: 'image_url';
  image_url: {
    url: string;
    detail?: string;
  };
}

export interface OpenAIFilePart {
  type: 'file';
  file: {
    filename?: string;
    file_data?: string;
    file_id?: string;
  };
}

export type OpenAIContentPart =
  | OpenAITextPart
  | OpenAIImageUrlPart
  | OpenAIFilePart;

export interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';

  content:
    | string
    | null
    | OpenAIContentPart[];

  tool_call_id?: string;

  tool_calls?: OpenAIToolCall[];
}

export interface ArnaruFileAttachment {
  buffer: Buffer;
  filename: string;
  mimeType: string;
}

export interface OpenAIChatRequest {
  model?: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  conversationId?: string;
  webSearch?: boolean;
  systemPrompt?: string;

  tools?: OpenAITool[];

  tool_choice?:
    | 'auto'
    | 'none'
    | 'required'
    | Record<string, unknown>;
}

export interface OpenAIChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;

  choices: Array<{
    index: number;

    message?: OpenAIMessage;

    delta?: {
      role?: string;
      content?: string | null;
      tool_calls?: OpenAIToolCall[];
    };

    finish_reason: string | null;
  }>;

  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface ArnaruSSEData {
  data?: string;
  answer?: string;
  text?: string;
  response?: string;
  message?: string;
  content?: string;
  delta?: string;
  token?: string;
  output?: string;
  result?: string;
  generated_text?: string;
  conversationId?: string;
  error?: string;

  choices?: Array<{
    delta?: {
      content?: string;
    };

    message?: {
      content?: string;
    };

    text?: string;
  }>;
}

export interface OpenAIModel {
  id: string;
  object: 'model';
  created: number;
  owned_by: string;
}

export interface OpenAIModelsResponse {
  object: 'list';
  data: OpenAIModel[];
}
