"""Resumable single-criterion repair followed by normal-mixture consolidation."""
from .criterion_sampling import BalancedSampler, decision

def failures(report, retention=False, baseline=None):
    def bad(m, minimum=.95):
        return m['accuracy']<minimum or m['format_accuracy']<.95 or (not retention and m['trace_accuracy']<.90)
    return [c for c,m in report['criteria'].items() if bad(m,max(.95,(baseline or {}).get(c,{}).get('accuracy',0)))
            or any(bad(s) for s in m.get('strata',{}).values())]

def add_strata(report, rows):
    for name,m in report['criteria'].items():
        labels={decision(r) for r in rows if r['criterion']==name and decision(r) is not None}
        m['strata']={}
        for label in sorted(labels):
            values=[s for r,s in zip(rows,report['samples']) if r['criterion']==name and decision(r)==label]
            m['strata'][label]={'examples':len(values),**{k:sum(s[v] for s in values)/len(values)
                for k,v in [('accuracy','answer_correct'),('trace_accuracy','trace_correct'),('format_accuracy','format_correct')]}}
    return report

class RepairController:
    def __init__(self, normal=120, repair=30, attempts=4, consolidation=3, state=None):
        self.normal=normal;self.repair=repair;self.attempts=attempts;self.consolidation=consolidation
        self.state=state or dict(mode='normal',focus=None,normal_done=0,repair_done=0,consolidation_done=0,streak=0,attempt_counts={})

    def focus(self, weak, retained):
        names=weak or retained
        if not names:return
        name=names[0];counts=self.state['attempt_counts'];counts[name]=counts.get(name,0)+1
        if counts[name]>self.attempts:raise RuntimeError('Repair attempt budget exhausted for '+name)
        self.state.update(mode='repair',focus=name,repair_done=0,streak=0)

    def observe(self, weak, retained):
        s=self.state;passed=not weak and not retained
        if s['mode']=='repair':
            s['repair_done']+=1;s['streak']=s['streak']+1 if s['focus'] not in weak+retained else 0
            if s['streak']>=2:s.update(mode='consolidate',consolidation_done=0,streak=0)
            elif s['repair_done']>=self.repair:self.focus(weak,retained)
            return False
        s['streak']=s['streak']+1 if passed else 0
        if s['mode']=='consolidate':
            s['consolidation_done']+=1
            if s['consolidation_done']<self.consolidation:return False
            if not passed:self.focus(weak,retained);return False
            return s['streak']>=2
        s['normal_done']+=1
        if s['normal_done']>=3 and s['streak']>=2:return True
        if s['normal_done']>=min(3,self.normal) and not passed:self.focus(weak,retained)
        return False

    def rows(self, active, prior, count, seed):
        if self.state['mode']=='repair':
            pool=[r for r in active+prior if r['criterion']==self.state['focus']]
            if not pool:raise ValueError('No training data for focused criterion')
            return BalancedSampler(pool).sample(count,seed)
        n=count*3//4 if prior else count
        return BalancedSampler(active).sample(n,seed)+BalancedSampler(prior).sample(count-n,seed+1)

class ScheduledRepairController(RepairController):
    """No attempt cutoff; full checks alone establish consecutive mastery."""
    def focus(self,weak,retained):
        names=weak or retained
        if not names:return
        name=names[0];counts=self.state['attempt_counts'];counts[name]=counts.get(name,0)+1
        self.state.update(mode='repair',focus=name,repair_done=0,streak=0)

    def due(self,round_,interval,maximum):
        return round_%interval==0 or self.state.get('full_pass_round')==round_-1 or round_==maximum

    def full_result(self,round_,weak,retained):
        if weak or retained:
            self.state.update(full_streak=0,full_pass_round=None)
            self.focus(weak,retained);return False
        previous=self.state.get('full_pass_round')
        self.state['full_streak']=self.state.get('full_streak',0)+1 if previous==round_-1 else 1
        self.state['full_pass_round']=round_
        if self.state['mode']=='repair':self.state.update(mode='consolidate',consolidation_done=0,streak=0)
        consolidated=self.state['consolidation_done']>=self.consolidation if self.state.get('focus') else self.state['normal_done']>=3
        return self.state['full_streak']>=2 and consolidated and self.state['mode']!='repair'
