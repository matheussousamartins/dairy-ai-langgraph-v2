# Message Service Contract (Implementation)

Date: 2026-04-15

## Goal

This service is the official source of truth for app messages (user chat history, feedback, and message persistence).
The AI backend remains responsible for streaming/orchestration/RAG.

## Integration rule with AI Backend

- The app `chatId` must be sent as `session_id` to the AI API.
- The AI API accepts `session_id`, `chat_id`, and `chatId`.
- Do not transform this identifier: use the same value end-to-end.

## Data model (Message Service)

Table: `messages`

- `id` UUID PK
- `chatId` UUID FK -> chats (required)
- `role` enum: `user | assistant` (required)
- `content` text (required)
- `attachments` jsonb nullable
- `source` enum: `curated | web` nullable (assistant only)
- `feedback` enum: `up | down` nullable
- `createdAt` timestamptz
- `deletedAt` timestamptz nullable (soft delete)

Attachment object:

```json
{
  "id": "uuid",
  "type": "string",
  "uri": "/uploads/file.ext",
  "name": "file.ext",
  "mimeType": "application/pdf"
}
```

## Required endpoints

## 1) Save user message

- `POST /messages/user`

Request:

```json
{
  "userId": "uuid",
  "chatId": "uuid",
  "content": "string",
  "attachments": [
    {
      "type": "string",
      "uri": "string",
      "name": "string",
      "mimeType": "string"
    }
  ]
}
```

Rules:
- validate ownership of `chatId` by `userId` (404 `CHAT_NOT_FOUND`)
- if this is the first message, set chat title (`content` truncated to 80 chars)
- generate UUID for each attachment
- persist message with `role=user`

Response `201`:

```json
{
  "message": {
    "id": "uuid",
    "chatId": "uuid",
    "role": "user",
    "content": "string",
    "attachments": [],
    "createdAt": "ISO8601"
  }
}
```

## 2) Save assistant message

- `POST /messages/assistant`

Request:

```json
{
  "chatId": "uuid",
  "content": "string",
  "source": "curated"
}
```

Rules:
- persist message with `role=assistant`
- persist `source` (`curated` or `web`)

Response `201`:

```json
{
  "message": {
    "id": "uuid",
    "chatId": "uuid",
    "role": "assistant",
    "content": "string",
    "attachments": [],
    "source": "curated",
    "feedback": null,
    "createdAt": "ISO8601"
  }
}
```

## 3) Get messages with cursor pagination

- `GET /messages?chatId=uuid&userId=uuid&limit=50&cursor=uuid`

Rules:
- validate ownership `chatId` + `userId` (404 `CHAT_NOT_FOUND`)
- order ASC by `createdAt`
- fetch `limit+1` to compute `hasMore`
- if `cursor` is provided, return messages where `createdAt > cursor.createdAt`

Response `200`:

```json
{
  "messages": [
    {
      "id": "uuid",
      "chatId": "uuid",
      "role": "user",
      "content": "string",
      "attachments": [],
      "source": null,
      "feedback": null,
      "createdAt": "ISO8601"
    }
  ],
  "nextCursor": "uuid",
  "hasMore": false
}
```

## 4) Submit feedback

- `POST /messages/feedback`

Request:

```json
{
  "userId": "uuid",
  "chatId": "uuid",
  "messageId": "uuid",
  "type": "up"
}
```

Rules:
- validate ownership `chatId` + `userId` (404 `CHAT_NOT_FOUND`)
- validate `messageId` within that chat (404 `MESSAGE_NOT_FOUND`)
- update `feedback`

Response: `204`

## 5) Delete messages by chat (soft delete)

- `DELETE /messages?chatId=uuid`

Rule:
- set `deletedAt` for all messages in the chat

Response: `204`

## 6) Feedback analytics

- `GET /messages/analytics/feedback`

Response `200`:

```json
{
  "up": 120,
  "down": 15,
  "total": 135
}
```

## Standard error codes

- `CHAT_NOT_FOUND` -> HTTP 404
- `MESSAGE_NOT_FOUND` -> HTTP 404

Recommended error shape:

```json
{
  "code": "CHAT_NOT_FOUND",
  "message": "Chat not found"
}
```

## Recommended production flow

1. App saves user message in Message Service (`/messages/user`).
2. App calls AI streaming endpoint (`/webhook/orquestrador/stream`) with `session_id=chatId`.
3. After stream completion, app saves assistant response in Message Service (`/messages/assistant`) with `source`.
4. Feedback is always sent to Message Service (`/messages/feedback`).
5. App chat history is read from `GET /messages`.

