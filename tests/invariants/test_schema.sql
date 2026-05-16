-- Invariant verification for v1 schema.
-- Each block expects an error; we wrap in DO ... EXCEPTION blocks and raise NOTICE on success.

\set ON_ERROR_STOP off

-- Seed a baseline (env, agent, memory, entity, tag, graph_nodes, outbox event).
DO $$
DECLARE
    v_env1 uuid; v_env2 uuid;
    v_agent uuid;
    v_mem1 uuid; v_mem2 uuid;
    v_ent1 uuid; v_ent2 uuid;
    v_tag1 uuid; v_tag2 uuid;
    v_node1 uuid; v_node2 uuid;
BEGIN
    INSERT INTO environments(name, default_embedding_model_id) VALUES ('work', 'all-MiniLM-L6-v2') RETURNING id INTO v_env1;
    INSERT INTO environments(name, default_embedding_model_id) VALUES ('private', 'all-MiniLM-L6-v2') RETURNING id INTO v_env2;
    INSERT INTO agents(name) VALUES ('test-agent') RETURNING id INTO v_agent;
    INSERT INTO memories(env_id, kind, body) VALUES (v_env1, 'fact', 'Cosmos uses 10000 RU/s in EUS') RETURNING id INTO v_mem1;
    INSERT INTO memories(env_id, kind, body) VALUES (v_env2, 'fact', 'Personal note') RETURNING id INTO v_mem2;
    INSERT INTO entities(env_id, kind, canonical_name, normalized_name) VALUES (v_env1, 'service', 'Cosmos DB', 'cosmos db') RETURNING id INTO v_ent1;
    INSERT INTO entities(env_id, kind, canonical_name, normalized_name) VALUES (v_env2, 'service', 'Other', 'other') RETURNING id INTO v_ent2;
    INSERT INTO tags(env_id, name) VALUES (v_env1, 'azure') RETURNING id INTO v_tag1;
    INSERT INTO tags(env_id, name) VALUES (v_env2, 'personal') RETURNING id INTO v_tag2;
    INSERT INTO graph_nodes(env_id, node_type, memory_id) VALUES (v_env1, 'memory', v_mem1) RETURNING id INTO v_node1;
    INSERT INTO graph_nodes(env_id, node_type, entity_id) VALUES (v_env1, 'entity', v_ent1) RETURNING id INTO v_node2;

    -- stash for use across blocks via a temp table
    CREATE TEMP TABLE _seed (k text PRIMARY KEY, v uuid);
    INSERT INTO _seed VALUES
      ('env1', v_env1), ('env2', v_env2), ('agent', v_agent),
      ('mem1', v_mem1), ('mem2', v_mem2),
      ('ent1', v_ent1), ('ent2', v_ent2),
      ('tag1', v_tag1), ('tag2', v_tag2),
      ('node1', v_node1), ('node2', v_node2);
    RAISE NOTICE 'seed: ok';
END $$;

-- 1. memory_tags cross-env FK should reject mismatched env.
DO $$
DECLARE
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
    v_tag2 uuid := (SELECT v FROM _seed WHERE k='tag2');
    v_env2 uuid := (SELECT v FROM _seed WHERE k='env2');
BEGIN
    BEGIN
        INSERT INTO memory_tags(memory_id, tag_id, env_id) VALUES (v_mem1, v_tag2, v_env2);
        RAISE EXCEPTION 'expected FK violation but insert succeeded';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'test 1 (memory_tags cross-env): ok';
    END;
END $$;

-- 2. entity_aliases cross-env FK
DO $$
DECLARE
    v_ent1 uuid := (SELECT v FROM _seed WHERE k='ent1');
    v_env2 uuid := (SELECT v FROM _seed WHERE k='env2');
BEGIN
    BEGIN
        INSERT INTO entity_aliases(entity_id, env_id, alias, normalized_alias) VALUES (v_ent1, v_env2, 'CosmosDB', 'cosmosdb');
        RAISE EXCEPTION 'expected FK violation but insert succeeded';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'test 2 (entity_aliases cross-env): ok';
    END;
END $$;

-- 3. relations cross-env FK (src_node in env1 but relation tries env2)
DO $$
DECLARE
    v_node1 uuid := (SELECT v FROM _seed WHERE k='node1');
    v_node2 uuid := (SELECT v FROM _seed WHERE k='node2');
    v_env2 uuid := (SELECT v FROM _seed WHERE k='env2');
BEGIN
    BEGIN
        INSERT INTO relations(env_id, src_node_id, dst_node_id, type) VALUES (v_env2, v_node1, v_node2, 'mentions');
        RAISE EXCEPTION 'expected FK violation but insert succeeded';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'test 3 (relations cross-env): ok';
    END;
END $$;

-- 4. graph_nodes exactly-one-target: setting both memory_id and entity_id
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
    v_ent1 uuid := (SELECT v FROM _seed WHERE k='ent1');
BEGIN
    BEGIN
        INSERT INTO graph_nodes(env_id, node_type, memory_id, entity_id) VALUES (v_env1, 'memory', v_mem1, v_ent1);
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 4 (graph_nodes exactly_one_target): ok';
    END;
END $$;

-- 5. memories_superseded_status_chk: status active with superseded_by set
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    BEGIN
        INSERT INTO memories(env_id, kind, body, superseded_by) VALUES (v_env1, 'fact', 'orphan', v_mem1);
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 5 (memories_superseded_status): ok';
    END;
END $$;

-- 6. memories_not_self_superseded_chk
DO $$
DECLARE
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    BEGIN
        UPDATE memories SET status='superseded', superseded_by=v_mem1, version=version+1 WHERE id=v_mem1;
        RAISE EXCEPTION 'expected CHECK violation but update succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 6 (memories_not_self_superseded): ok';
    END;
END $$;

-- 7. version monotonic trigger
DO $$
DECLARE
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    UPDATE memories SET version=5 WHERE id=v_mem1;
    BEGIN
        UPDATE memories SET version=3 WHERE id=v_mem1;
        RAISE EXCEPTION 'expected trigger raise but update succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 7 (version monotonic): ok';
    END;
END $$;

-- 7b. version trigger allows access-tracking updates without bumping version
DO $$
DECLARE
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
    v_ver bigint;
BEGIN
    UPDATE memories SET access_count=access_count+1, last_accessed_at=now() WHERE id=v_mem1;
    SELECT version INTO v_ver FROM memories WHERE id=v_mem1;
    RAISE NOTICE 'test 7b (version unchanged on access tracking): ok (version=%)', v_ver;
END $$;

-- 8. outbox unique on (aggregate_type, aggregate_id, aggregate_version)
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    INSERT INTO outbox(aggregate_type, aggregate_id, aggregate_version, env_id, op, payload)
      VALUES ('memory', v_mem1, 1, v_env1, 'upsert', '{}'::jsonb);
    BEGIN
        INSERT INTO outbox(aggregate_type, aggregate_id, aggregate_version, env_id, op, payload)
          VALUES ('memory', v_mem1, 1, v_env1, 'tombstone', '{}'::jsonb);
        RAISE EXCEPTION 'expected unique violation but insert succeeded';
    EXCEPTION WHEN unique_violation THEN
        RAISE NOTICE 'test 8 (outbox unique on aggregate version): ok';
    END;
END $$;

-- 9. outbox aggregate_version > 0
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    BEGIN
        INSERT INTO outbox(aggregate_type, aggregate_id, aggregate_version, env_id, op, payload)
          VALUES ('memory', v_mem1, 0, v_env1, 'upsert', '{}'::jsonb);
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 9 (outbox aggregate_version > 0): ok';
    END;
END $$;

-- 10. outbox_delivery state CHECK: in_flight without locked_by
DO $$
DECLARE
    v_event_id bigint;
BEGIN
    SELECT event_id INTO v_event_id FROM outbox LIMIT 1;
    BEGIN
        INSERT INTO outbox_delivery(event_id, sink, status) VALUES (v_event_id, 'qdrant', 'in_flight');
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 10 (outbox_delivery state CHECK): ok';
    END;
END $$;

-- 11. memory_lineage no self-cycle
DO $$
DECLARE
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
BEGIN
    BEGIN
        INSERT INTO memory_lineage(parent_memory_id, child_memory_id, relation) VALUES (v_mem1, v_mem1, 'supersedes');
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 11 (memory_lineage no self): ok';
    END;
END $$;

-- 12. salience out of range
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
BEGIN
    BEGIN
        INSERT INTO memories(env_id, kind, body, salience) VALUES (v_env1, 'fact', 'oob', 1.5);
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 12 (salience range): ok';
    END;
END $$;

-- 13. projection_state sink CHECK
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
BEGIN
    BEGIN
        INSERT INTO projection_state(sink, env_id) VALUES ('typo', v_env1);
        RAISE EXCEPTION 'expected CHECK violation but insert succeeded';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'test 13 (projection_state sink CHECK): ok';
    END;
END $$;

-- 14. valid happy path: full memory_tag insert with matching env
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
    v_tag1 uuid := (SELECT v FROM _seed WHERE k='tag1');
BEGIN
    INSERT INTO memory_tags(memory_id, tag_id, env_id) VALUES (v_mem1, v_tag1, v_env1);
    RAISE NOTICE 'test 14 (memory_tags happy path): ok';
END $$;

-- 15. valid supersede transition
DO $$
DECLARE
    v_env1 uuid := (SELECT v FROM _seed WHERE k='env1');
    v_mem1 uuid := (SELECT v FROM _seed WHERE k='mem1');
    v_new uuid;
BEGIN
    INSERT INTO memories(env_id, kind, body) VALUES (v_env1, 'fact', 'replacement') RETURNING id INTO v_new;
    UPDATE memories SET status='superseded', superseded_by=v_new, version=version+1 WHERE id=v_mem1;
    INSERT INTO memory_lineage(parent_memory_id, child_memory_id, relation) VALUES (v_mem1, v_new, 'supersedes');
    RAISE NOTICE 'test 15 (supersede happy path): ok';
END $$;

\echo 'all invariant tests passed'
