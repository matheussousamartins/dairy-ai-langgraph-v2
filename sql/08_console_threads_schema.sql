create table if not exists public.console_threads (
  id text primary key,
  owner_id text not null,
  title text not null default 'Nova sessão',
  preview text not null default '',
  message_count integer not null default 0,
  question text not null default '',
  last_agent_id text null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists console_threads_owner_updated_idx
  on public.console_threads (owner_id, updated_at desc);

create table if not exists public.console_thread_messages (
  id text primary key,
  thread_id text not null references public.console_threads(id) on delete cascade,
  owner_id text not null,
  type text not null check (type in ('human', 'ai')),
  content text not null default '',
  turn_id text null,
  response_metadata jsonb not null default '{}'::jsonb,
  tool_calls jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists console_thread_messages_thread_created_idx
  on public.console_thread_messages (thread_id, created_at asc);

create index if not exists console_thread_messages_owner_thread_idx
  on public.console_thread_messages (owner_id, thread_id);

create table if not exists public.console_thread_traces (
  id text primary key,
  thread_id text not null references public.console_threads(id) on delete cascade,
  owner_id text not null,
  turn_id text not null,
  response_metadata jsonb not null default '{}'::jsonb,
  trace jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists console_thread_traces_owner_thread_idx
  on public.console_thread_traces (owner_id, thread_id);

create index if not exists console_thread_traces_turn_idx
  on public.console_thread_traces (thread_id, turn_id);

create or replace function public.set_console_threads_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

drop trigger if exists trg_console_threads_updated_at on public.console_threads;
create trigger trg_console_threads_updated_at
before update on public.console_threads
for each row
execute function public.set_console_threads_updated_at();
