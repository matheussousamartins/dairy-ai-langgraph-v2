-- Restore per-agent embedding tables from backup tables.
--
-- Safe behavior:
-- - Does not drop current generic tables (embeddings_especialista / embeddings_regulatorios).
-- - Creates each expected per-agent table only if it does not exist.
-- - Inserts backup rows only when the target table is empty.
--
-- Run after checking the _bkp_embeddings_agente_* tables exist in Supabase.

begin;

create table if not exists public.embeddings_agente_0_base_geral
  (like public._bkp_embeddings_agente_0 including all);
insert into public.embeddings_agente_0_base_geral
select * from public._bkp_embeddings_agente_0
where not exists (select 1 from public.embeddings_agente_0_base_geral);

create table if not exists public.embeddings_agente_1_queijos
  (like public._bkp_embeddings_agente_1 including all);
insert into public.embeddings_agente_1_queijos
select * from public._bkp_embeddings_agente_1
where not exists (select 1 from public.embeddings_agente_1_queijos);

create table if not exists public.embeddings_agente_2_fermentados
  (like public._bkp_embeddings_agente_2 including all);
insert into public.embeddings_agente_2_fermentados
select * from public._bkp_embeddings_agente_2
where not exists (select 1 from public.embeddings_agente_2_fermentados);

create table if not exists public.embeddings_agente_3_regulatorios
  (like public._bkp_embeddings_agente_3 including all);
insert into public.embeddings_agente_3_regulatorios
select * from public._bkp_embeddings_agente_3
where not exists (select 1 from public.embeddings_agente_3_regulatorios);

create table if not exists public.embeddings_agente_4_qualidade_leite
  (like public._bkp_embeddings_agente_4 including all);
insert into public.embeddings_agente_4_qualidade_leite
select * from public._bkp_embeddings_agente_4
where not exists (select 1 from public.embeddings_agente_4_qualidade_leite);

create table if not exists public.embeddings_agente_5_defeitos
  (like public._bkp_embeddings_agente_5 including all);
insert into public.embeddings_agente_5_defeitos
select * from public._bkp_embeddings_agente_5
where not exists (select 1 from public.embeddings_agente_5_defeitos);

create table if not exists public.embeddings_agente_6_formulacao
  (like public._bkp_embeddings_agente_6 including all);
insert into public.embeddings_agente_6_formulacao
select * from public._bkp_embeddings_agente_6
where not exists (select 1 from public.embeddings_agente_6_formulacao);

commit;

select 'embeddings_agente_0_base_geral' as table_name, count(*) as rows from public.embeddings_agente_0_base_geral
union all
select 'embeddings_agente_1_queijos', count(*) from public.embeddings_agente_1_queijos
union all
select 'embeddings_agente_2_fermentados', count(*) from public.embeddings_agente_2_fermentados
union all
select 'embeddings_agente_3_regulatorios', count(*) from public.embeddings_agente_3_regulatorios
union all
select 'embeddings_agente_4_qualidade_leite', count(*) from public.embeddings_agente_4_qualidade_leite
union all
select 'embeddings_agente_5_defeitos', count(*) from public.embeddings_agente_5_defeitos
union all
select 'embeddings_agente_6_formulacao', count(*) from public.embeddings_agente_6_formulacao
order by table_name;
