#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Endpoint {
    Messages,
    Responses,
    ChatCompletions,
    Embeddings,
    Rerank,
    AudioSpeech,
    AudioTranscriptions,
    ImagesGenerations,
    ImagesEdits,
}

pub const SUPPORTED_ENDPOINT_PATHS: [&str; 9] = [
    "/v1/messages",
    "/v1/responses",
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/images/generations",
    "/v1/images/edits",
];

impl Endpoint {
    pub fn from_path(path: &str) -> Option<Self> {
        match path {
            "/v1/messages" => Some(Self::Messages),
            "/v1/responses" => Some(Self::Responses),
            "/v1/chat/completions" => Some(Self::ChatCompletions),
            "/v1/embeddings" => Some(Self::Embeddings),
            "/v1/rerank" => Some(Self::Rerank),
            "/v1/audio/speech" => Some(Self::AudioSpeech),
            "/v1/audio/transcriptions" => Some(Self::AudioTranscriptions),
            "/v1/images/generations" => Some(Self::ImagesGenerations),
            "/v1/images/edits" => Some(Self::ImagesEdits),
            _ => None,
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "messages" => Some(Self::Messages),
            "responses" => Some(Self::Responses),
            "chat_completions" => Some(Self::ChatCompletions),
            "embeddings" => Some(Self::Embeddings),
            "rerank" => Some(Self::Rerank),
            "audio_speech" => Some(Self::AudioSpeech),
            "audio_transcriptions" => Some(Self::AudioTranscriptions),
            "images_generations" => Some(Self::ImagesGenerations),
            "images_edits" => Some(Self::ImagesEdits),
            _ => None,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::AudioSpeech => "audio_speech",
            Self::AudioTranscriptions => "audio_transcriptions",
            Self::ImagesGenerations => "images_generations",
            Self::ImagesEdits => "images_edits",
        }
    }

    pub fn path(self) -> &'static str {
        match self {
            Self::Messages => "/v1/messages",
            Self::Responses => "/v1/responses",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Embeddings => "/v1/embeddings",
            Self::Rerank => "/v1/rerank",
            Self::AudioSpeech => "/v1/audio/speech",
            Self::AudioTranscriptions => "/v1/audio/transcriptions",
            Self::ImagesGenerations => "/v1/images/generations",
            Self::ImagesEdits => "/v1/images/edits",
        }
    }
}
