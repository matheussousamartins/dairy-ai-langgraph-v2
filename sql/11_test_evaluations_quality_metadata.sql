-- Adds production-quality evaluation metadata for agent improvement loops.
-- Safe/idempotent: only adds columns and indexes when missing.

alter table public.test_evaluations
  add column if not exists evaluator_id text null,
  add column if not exists environment text not null default 'test',
  add column if not exists app_version text null,
  add column if not exists git_sha text null,
  add column if not exists rag_architecture text null,
  add column if not exists prompt_version text null,
  add column if not exists retrieval_config_version text null,
  add column if not exists error_category text null,
  add column if not exists expected_answer text null,
  add column if not exists status text not null default 'new',
  add column if not exists answer_source text null,
  add column if not exists chosen_agent_ids jsonb not null default '[]'::jsonb,
  add column if not exists primary_agent_id text null,
  add column if not exists top_rag_score numeric(10,6) null,
  add column if not exists rag_sources jsonb not null default '[]'::jsonb,
  add column if not exists rag_search_count integer not null default 0,
  add column if not exists node_count integer not null default 0,
  add column if not exists latency_ms integer null,
  add column if not exists web_fallback_used boolean not null default false;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'test_evaluations_error_category_check'
      and conrelid = 'public.test_evaluations'::regclass
  ) then
    alter table public.test_evaluations
      add constraint test_evaluations_error_category_check
      check (
        error_category is null or error_category in (
          'retrieval',
          'routing',
          'consolidation',
          'hallucination',
          'missing_kb',
          'regulatory_conflict',
          'wrong_scope',
          'ui',
          'other'
        )
      );
  end if;

  if not exists (
    select 1
    from pg_constraint
    where conname = 'test_evaluations_status_quality_check'
      and conrelid = 'public.test_evaluations'::regclass
  ) then
    alter table public.test_evaluations
      add constraint test_evaluations_status_quality_check
      check (status in ('new', 'accepted', 'triaged', 'fixed', 'ignored', 'regression_test_added'));
  end if;
end $$;

create index if not exists idx_test_evaluations_status_created_at
  on public.test_evaluations(status, created_at desc);

create index if not exists idx_test_evaluations_error_category
  on public.test_evaluations(error_category);

create index if not exists idx_test_evaluations_environment_verdict
  on public.test_evaluations(environment, verdict, created_at desc);

create index if not exists idx_test_evaluations_metadata_gin
  on public.test_evaluations using gin (metadata jsonb_path_ops);

comment on column public.test_evaluations.metadata is
'Structured evaluation context: trace summary, RAG sources, runtime config, and training signals.';

comment on column public.test_evaluations.error_category is
'Human triage category for partial/incorrect answers.';

comment on column public.test_evaluations.expected_answer is
'Optional corrected answer or expected answer supplied by the evaluator.';
