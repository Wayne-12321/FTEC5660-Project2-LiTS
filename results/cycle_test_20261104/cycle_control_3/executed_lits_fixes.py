"""Finite-action BlocksWorld fixes, without modifying the frozen upstream checkout.

No reference plans, goal-distance oracle, transition replacement or cross-run cache.
"""
import logging
import math
from lits.components.policy.env_grounded import EnvGroundedPolicy
from lits.components.registry import register_policy
from lits.structures.env_grounded import EnvStep, EnvAction
from lits.agents.registry import register_search
from lits.agents.tree.mcts import MCTSSearch, MCTSConfig, _expand
from lits.agents.tree.common import _world_modeling

log = logging.getLogger(__name__)

@register_policy('blocksworld_unique', task_type='env_grounded')
class UniquePolicy(EnvGroundedPolicy):
    def __init__(self, *args, **kwargs):
        # Registry derives prompt keys from class names; keep upstream BW prompt.
        from lits_benchmark.blocksworld import blocksworld_policy_usr_prompt
        kwargs.setdefault('usr_prompt_spec', blocksworld_policy_usr_prompt())
        super().__init__(*args, **kwargs)

    def _get_actions(self, state, n_actions, temperature, query=None, **kwargs):
        excluded = set(getattr(self, '_existing_actions', ()))
        remaining = list(dict.fromkeys(self.generate_all_actions(state.env_state)))
        remaining = [a for a in remaining if a not in excluded]
        steps = []
        for _ in range(min(n_actions, len(remaining))):
            # A singleton has no choice to make; no model call is needed.
            if len(remaining) == 1 or self.base_model is None:
                action = remaining[0]
            else:
                prompt = self._build_prompt(state.env_state, query, remaining)
                response = self._call_model(prompt, temperature=temperature).text
                action = self._parse_llm_response(response, remaining)
                if action not in remaining:
                    log.warning('Invalid finite-action selection; using first legal remaining action')
                    action = remaining[0]
            steps.append(EnvStep(action=EnvAction(action)))
            remaining.remove(action)
        return steps

class FrontierExhausted(Exception):
    pass

@register_search('mcts_frontier', config_class=MCTSConfig)
class FrontierMCTS(MCTSSearch):
    def _open(self, node):
        # Depth-limit edges still require one transition to check for a goal.
        if node.state is None:
            return True
        if node.is_terminal or (self.config.force_terminating_on_depth_limit and
                                node.depth >= self.config.max_steps):
            return False
        return (not getattr(node, '_fully_expanded', False) or
                any(self._open(c) for c in node.children))

    def _do_select_path(self, query, query_idx, iteration):
        node, path = self.root, [self.root]
        if not self._open(node):
            raise FrontierExhausted()
        while node.state is not None and getattr(node, '_fully_expanded', False):
            candidates = [c for c in node.children if self._open(c)]
            unknown = [c for c in candidates if c.state is None]
            if unknown:
                node = max(unknown, key=lambda c: c.fast_reward)
            else:
                def uct(c):
                    visits = c.visit_count or len(c.cum_rewards)
                    return (float('inf') if not visits else c.Q + self.config.w_exp *
                            math.sqrt(math.log(max(2, node.visit_count or len(node.cum_rewards))) / visits))
                node = max(candidates, key=uct)
            path.append(node)
        # Upstream checks the depth cap before transitioning a selected node.
        # Select its parent so rollout evaluates this last legal edge exactly once.
        if (node.state is None and self.config.force_terminating_on_depth_limit
                and node.depth >= self.config.max_steps):
            path.pop()
            node = path[-1]
        log.info('FRONTIER iteration=%s selected_node=%s depth=%s', iteration, node.id, node.depth)
        return path

    def _do_expand(self, query, query_idx, node, policy, n_actions, **kwargs):
        # Preserve rollout siblings; upstream otherwise truncates them to rollout width.
        for child in node.children:
            child.from_expand = True
        previous = getattr(policy, '_existing_actions', ())
        policy._existing_actions = [c.action.action_str for c in node.children]
        try:
            _expand(query, query_idx, node, policy, n_actions, **kwargs)
        finally:
            policy._existing_actions = previous
        if kwargs.get('from_phase') == 'expand':
            node._fully_expanded = True
        legal_count = len(set(policy.generate_all_actions(node.state.env_state)))
        if len(node.children) >= legal_count:
            node._fully_expanded = True

    def _after_model(self, node):
        pass

    def _do_simulate(self, query, query_idx, path, config, world_model, policy,
                     reward_model, roll_out_steps=10000, on_step=None, **kwargs):
        for _ in range(roll_out_steps):
            node = path[-1]
            if not self._open(node):
                break
            self._do_expand(query, query_idx, node, policy, config.n_action_for_simulate,
                            reward_model=reward_model, world_model=world_model,
                            from_phase='simulate', **kwargs)
            candidates = [c for c in node.children if self._open(c)]
            if not candidates and not getattr(node, '_fully_expanded', False):
                self._do_expand(query, query_idx, node, policy, policy.n_actions,
                                reward_model=reward_model, world_model=world_model,
                                from_phase='expand', **kwargs)
                candidates = [c for c in node.children if self._open(c)]
            if not candidates:
                break
            child = candidates[config.simulate_choice([c.fast_reward for c in candidates])]
            child.is_simulated = True
            if on_step:
                on_step(child)
            _world_modeling(query, query_idx, child, transition_model=world_model,
                            reward_model=reward_model, from_phase='simulate')
            path.append(child)
            self._after_model(child)
            if child.is_terminal or (config.force_terminating_on_depth_limit and child.depth >= config.max_steps):
                break
        return False, []

    def search(self, query, query_idx):
        if self.config.add_continuation:
            raise ValueError('Frontier fix supports the continuation-disabled BlocksWorld configs only')
        try:
            return super().search(query, query_idx)
        except RuntimeError:
            # Preserve partial evidence on budget exhaustion, then keep failure status.
            self.save_full_tree(query_idx)
            raise
        except FrontierExhausted:
            log.info('FRONTIER exhausted: all configured candidate branches explored')
            self.save_full_tree(query_idx)
            return self._fallback_result(query, query_idx)

@register_search('mcts_cycle', config_class=MCTSConfig)
class CycleMCTS(FrontierMCTS):
    """Path-local cycle pruning after LLM transitions; all model roles retained."""
    def _setup(self, query, query_idx):
        super()._setup(query, query_idx)
        self._cycle_events=[]
        self._cycle_parse_failures=set()

    def _cyclic(self, node):
        if getattr(node,'_cycle_pruned',False):return True
        if node.state is None or node.parent is None or node.is_terminal:return False
        from cycle_state import canonical_state
        root=node
        while root.parent is not None:root=root.parent
        initial_key=canonical_state(root.state.env_state)
        if initial_key is None:return False
        blocks={b for b,s in initial_key}
        key=canonical_state(node.state.env_state,blocks)
        if key is None:
            self._cycle_parse_failures.add(node.id)
            return False
        ancestor=node.parent
        while ancestor is not None:
            if ancestor.state is not None and canonical_state(ancestor.state.env_state,blocks)==key:
                node._cycle_pruned=True
                self._cycle_events.append(dict(node_id=node.id,ancestor_id=ancestor.id,
                    depth=node.depth,state_key=key))
                log.info('CYCLE prune node=%s ancestor=%s depth=%s',node.id,ancestor.id,node.depth)
                return True
            ancestor=ancestor.parent
        return False

    def _after_model(self, node):
        self._cyclic(node)

    def _open(self, node):
        if self._cyclic(node):return False
        return super()._open(node)

    def _do_expand(self, query, query_idx, node, policy, n_actions, **kwargs):
        if self._cyclic(node):return
        return super()._do_expand(query,query_idx,node,policy,n_actions,**kwargs)

    def search(self, query, query_idx):
        try:
            return super().search(query,query_idx)
        finally:
            from common import write
            from pathlib import Path
            if self.checkpoint_dir:
                write(Path(self.checkpoint_dir)/'cycle_audit.json',dict(
                    root_parse_valid=__import__('cycle_state').canonical_state(self.root.state.env_state) is not None,
                    pruned_nodes=self._cycle_events,parse_failure_node_ids=sorted(self._cycle_parse_failures),
                    method='LLM states compared only with ancestors; invalid/incomplete text is not pruned'))
