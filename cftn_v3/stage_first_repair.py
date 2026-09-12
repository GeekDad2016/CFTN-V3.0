"""Learn the whole stage before bounded repair and another normal block."""
from .criterion_repair import ScheduledRepairController
from .criterion_sampling import BalancedSampler

class StageFirstController(ScheduledRepairController):
    def focus(self,weak,retained):
        if not (weak or retained):return
        if self.state.get('recovery_blocks',0)>=self.attempts:
            raise RuntimeError('Normal and recovery cycle budget exhausted')
        super().focus(weak,retained)
        if self.state.get('subskill_recovery') and self.state.get('pending_subskills'):
            self.state.update(recovery_queue=list(self.state['pending_subskills']),
                              focus=self.state['pending_subskills'][0],recovery_gate_streak=0)
        self.state['focus_streak']=0
        self.state['recovery_blocks']=self.state.get('recovery_blocks',0)+1

    def observe(self,weak,retained):
        s=self.state;passed=not weak and not retained
        s['streak']=s['streak']+1 if passed else 0
        if s['mode']=='repair' and s.get('recovery_queue'):
            s['repair_done']+=1;s['recovery_total']=s.get('recovery_total',0)+1
            s['focus_streak']=s.get('focus_streak',0)+1 if s['focus'] not in weak+retained else 0
            # Queue advances only after its full focused held-out panel passes twice.
            return False
        if s['mode']=='repair':
            s['repair_done']+=1;s['recovery_total']=s.get('recovery_total',0)+1
            s['focus_streak']=s.get('focus_streak',0)+1 if s['focus'] not in weak+retained else 0
            if s['focus_streak']>=2 or s['repair_done']>=self.repair:
                s.update(mode='normal',normal_done=0,streak=0,full_streak=0,full_pass_round=None,focus_streak=0)
        else:
            s['normal_done']+=1;s['normal_total']=s.get('normal_total',0)+1
        return False

    def due(self,round_,interval,maximum):
        s=self.state
        if s.get('recovery_queue'):return False
        return (round_%interval==0 or round_==maximum or s.get('full_pass_round')==round_-1
                or s['mode']=='normal' and (s['normal_done']>=self.normal or s['normal_done']>=self.consolidation and s['streak']>=2))

    def next_check(self,round_,interval,maximum):
        if self.due(round_,interval,maximum):return round_
        return min(maximum,(round_//interval+1)*interval)

    def full_result(self,round_,weak,retained):
        s=self.state
        if weak or retained:
            s.update(full_streak=0,full_pass_round=None)
            if s['mode']=='normal' and s['normal_done']>=self.normal:self.focus(weak,retained)
            return False
        previous=s.get('full_pass_round');s['full_streak']=s.get('full_streak',0)+1 if previous==round_-1 else 1
        s['full_pass_round']=round_
        if s['mode']=='repair':
            s.update(mode='normal',normal_done=0,streak=0,full_streak=0,full_pass_round=None)
            return False
        return s['normal_done']>=self.consolidation and s['full_streak']>=2

    def recovery_result(self,weak):
        s=self.state
        s['recovery_gate_streak']=s.get('recovery_gate_streak',0)+1 if s['focus'] not in weak else 0
        if s['recovery_gate_streak']>=2:
            s['recovery_queue'].pop(0);s['recovery_gate_streak']=0;s['repair_done']=0
            if s['recovery_queue']:s['focus']=s['recovery_queue'][0]
            else:s.update(mode='normal',focus=None,normal_done=0,streak=0,full_streak=0,full_pass_round=None)
        elif s['repair_done']>=240:
            raise RuntimeError('Targeted recovery did not pass within 240 rounds; inspect recovery evidence')

    def rows(self,active,prior,count,seed):
        if self.state['mode']!='repair':return super().rows(active,prior,count,seed)
        if self.state.get('subskill_recovery') and self.state.get('recovery_queue'):
            from .subskill_recovery import sample
            return sample(active,prior,self.state['focus'],count,seed)
        focus=self.state['focus'];pool=[r for r in active+prior if r['criterion']==focus]
        replay=[r for r in active+prior if r['criterion']!=focus]
        if self.state.get('recovery_queue'):
            # Equal weight for original/direct cases and explicit scaffold cases.
            fresh=[r for r in pool if r.get('source_record')=='arithmetic_recovery_v1']
            pool=fresh or pool
        n=count*4//5 if replay else count
        return BalancedSampler(pool).sample(n,seed)+BalancedSampler(replay).sample(count-n,seed+1)
