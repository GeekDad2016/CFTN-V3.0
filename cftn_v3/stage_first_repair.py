"""Learn the whole stage before bounded repair and another normal block."""
from .criterion_repair import ScheduledRepairController
from .criterion_sampling import BalancedSampler

class StageFirstController(ScheduledRepairController):
    def focus(self,weak,retained):
        if not (weak or retained):return
        if self.state.get('recovery_blocks',0)>=self.attempts:
            raise RuntimeError('Normal and recovery cycle budget exhausted')
        super().focus(weak,retained)
        self.state['focus_streak']=0
        self.state['recovery_blocks']=self.state.get('recovery_blocks',0)+1

    def observe(self,weak,retained):
        s=self.state;passed=not weak and not retained
        s['streak']=s['streak']+1 if passed else 0
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

    def rows(self,active,prior,count,seed):
        if self.state['mode']!='repair':return super().rows(active,prior,count,seed)
        focus=self.state['focus'];pool=[r for r in active+prior if r['criterion']==focus]
        replay=[r for r in active+prior if r['criterion']!=focus]
        n=count*4//5 if replay else count
        return BalancedSampler(pool).sample(n,seed)+BalancedSampler(replay).sample(count-n,seed+1)
