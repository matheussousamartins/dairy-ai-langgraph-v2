-- Persistência da camada de testes do console
-- Objetivo: manter avaliações desacopladas do backend dos agentes do cliente

create extension if not exists pgcrypto;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

create table if not exists public.test_sessions (
  id uuid primary key default gen_random_uuid(),
  thread_id text not null,
  thread_title text not null,
  status text not null default 'active' check (status in ('active', 'completed')),
  source text not null default 'console',
  evaluated_count integer not null default 0,
  correct_count integer not null default 0,
  partial_count integer not null default 0,
  incorrect_count integer not null default 0,
  score_percent integer not null default 0,
  started_at timestamptz not null default timezone('utc', now()),
  ended_at timestamptz null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.test_evaluations (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.test_sessions(id) on delete cascade,
  thread_id text not null,
  message_id text not null,
  turn_id text null,
  verdict text not null check (verdict in ('correct', 'partial', 'incorrect')),
  score numeric(4,2) not null check (score in (0, 0.5, 1)),
  question text not null,
  answer text not null,
  agent_id text null,
  model_id text null,
  comment text null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  constraint uq_test_eval_session_message unique (session_id, message_id)
);

create index if not exists idx_test_sessions_thread_id
  on public.test_sessions(thread_id);

create index if not exists idx_test_sessions_status_updated_at
  on public.test_sessions(status, updated_at desc);

create index if not exists idx_test_evaluations_session_id
  on public.test_evaluations(session_id);

create index if not exists idx_test_evaluations_thread_id_created_at
  on public.test_evaluations(thread_id, created_at desc);

create index if not exists idx_test_evaluations_turn_id
  on public.test_evaluations(turn_id);

create index if not exists idx_test_evaluations_verdict
  on public.test_evaluations(verdict);

drop trigger if exists trg_test_sessions_updated_at on public.test_sessions;
create trigger trg_test_sessions_updated_at
before update on public.test_sessions
for each row
execute function public.set_updated_at();

drop trigger if exists trg_test_evaluations_updated_at on public.test_evaluations;
create trigger trg_test_evaluations_updated_at
before update on public.test_evaluations
for each row
execute function public.set_updated_at();

create or replace view public.v_test_session_quality_daily as
select
  date_trunc('day', started_at)::date as day,
  count(*) as sessions_total,
  avg(score_percent)::numeric(10,2) as avg_score_percent,
  sum(correct_count) as correct_total,
  sum(partial_count) as partial_total,
  sum(incorrect_count) as incorrect_total
from public.test_sessions
group by 1
order by 1 desc;

comment on table public.test_sessions is
'Sessões de teste do console, desacopladas do backend dos agentes do cliente.';

comment on table public.test_evaluations is
'Avaliações por resposta do agente (correta, parcial, incorreta) com observação opcional.';
