# Standalone specification of an online controller

“v60” is a label for the reference algorithm defined below (five draw rows,
no posterior parameter sampling, 200 particles). The specification is
self-contained. No source tree, prior agent, task glossary, or external data
is needed. Sections 1–8 define behavior; §9 supplies numerical checks.

## Review brief (not part of the algorithm)

**Problem.** At each step a learner receives a vector of anonymous sensor
values, with one distinguished utility coordinate, and chooses one action
from a finite set. It then observes the next vector and learns online. The
environment may contain delayed effects, partially observed dynamics, and
abrupt changes. The learner receives no task identifier, latent state,
future observation, or externally supplied sensor role. The aim of a
redesign is to maintain or improve utility over repeated interactions,
including adaptation after a change, at lower computational or structural
cost. Neither the two-step planning score nor a particular benchmark is the
ultimate objective. The algorithm below is the reference, not a list of
mechanisms a redesign must keep.

**Review request.** Find the smallest changes with the largest justified
benefit. Work from the equations and operation order, not from the names of
the components. In particular, look for repeated sufficient statistics,
reductions that can be eliminated, decisions invariant to parts of their
input, and dependencies discarded when one calculation feeds another.
Separate proposals into:

- **Exact simplifications:** specify the before/after calculation, the
  assumptions under which they are identical, and whether equality is
  algebraic, action-level with a fixed seed, or bitwise. Preserve the
  public transition and RNG semantics if claiming identical actions.
- **Behavioral changes:** give a concrete case where the reference and the
  replacement differ, explain which existing rule is replaced, and give a
  minimal case that would falsify the claimed benefit. Preserve the online
  interface and do not add environment labels or privileged observations.

For each proposal state what can be deleted, effects on worst-case and
common-case work and retained state, and the simplest meaningful check.
The input sequences in §9 check implementation agreement, not policy
quality. If the specification allows multiple implementations with
different actions, report the first ambiguous step instead of guessing.
Independent implementation parity has not been certified.

## Algorithm

1. [Conventions and public transition](#1-conventions-and-public-transition)
2. [Initial state](#2-initial-state)
3. [Process bank](#3-process-bank)
4. [Sensory law](#4-sensory-law)
5. [Reward learning](#5-reward-learning)
6. [Reward readout](#6-reward-readout)
7. [Action values](#7-action-values)
8. [Real sensory update](#8-real-sensory-update)
9. [Numerical conformance](#9-numerical-conformance)

## 1. Conventions and public transition

Input parameters: observation width `n`, action count `A≥1`, value coordinate
`v`, nonnegative integer seed `s`; `J=n−1≥1`. Sensor coordinates `ch` are the
indices other than `v` in ascending order. Inputs are finite vectors in
`[0,1]^n`. Actions and array indices are zero-based. Define `bits(x)=[x[ch]>.5]`.
For a real transition `x,a,y`, write `b=bits(x)`, `b'=bits(y)`, `r=y[v]`.
`h_l(b,H)` is `b` if `l=0`, the `l`-th element from the end of list `H` if it
exists, and zero otherwise. `H` contains past origins, oldest first. Write
`B(u,p)=p if u else 1−p`, `norm(x)=x/sum(x)`, `softmax(x)=norm(exp(x−max(x)))`,
`clip(x,l,u)=max(l,min(u,x))`, `LSE` for log-sum-exp, and `I(condition)` for
its zero/one indicator. Array products in these expressions are elementwise;
`@` denotes contraction over the shared dimension. Logs are natural. Slices
and loops use upper-exclusive endpoints unless stated otherwise.

Use a controller RNG `default_rng(SeedSequence([s,541]))` and a factor RNG
`default_rng(s)`. All RNG operations in this document on `default_rng` are
NumPy Generator operations (see §9). Set `pending = none`.

```text
step(x):
    require pending = none
    U ← controller_rng.random((5,J+4))             # row-major
    Q ← action_values(x,U)                         # §7, no state change
    Amax ← ascending list of a such that Q[a] == max(Q)
    a ← controller_rng.choice(Amax)                # call even for singleton
    prepare_real_prediction(x,a)                   # §4.5
    pending ← (x,a)
    return a
learn(y):
    require pending exists
    (x,a) ← pending
    reward_learn(bits(x),H,a,y[v],bits(y))            # §5, before H changes
    sensory_learn(x,a,y)                            # §8
    pending ← none
```

## 2. Initial state

All arrays and counts specified below start at zero unless otherwise stated.
`none`, empty lists, empty dictionaries, and booleans are specified separately.
Do not share RNGs between banks. Use ascending order for unordered indices
where not specified otherwise; resolve `argmax` by the first maximum.

**Factor.** `N=200`, `K=10`, `k_parent=5`, history cap 5, candidate lags
`0,…,4`, own lags `0,1,2`. Set `m=3+A+5`. Enumerate the subsets of feature
positions as empty, all singletons, all pairs in lexicographic order; `S`
is the number of subsets and `Fcells=1+2m+4*m(m−1)/2`. Concatenate their cells
in this order, numbering within a subset by the binary code with its first
feature least significant. Pooled counts `g0,g1[J,Fcells]`, pooled table log
scores `eg[J,S]`; particle counts `q0,q1[N,K,J,Fcells]`, log scores
`ev[N,K,J,S+1]` and equal-shaped weights `Ew=1`. The latter four arrays are
binary32. Additive coefficients `mu[N,K,J,m+1]`, `mu_shared[J,m+1]` and
covariances of corresponding shape with an extra dimension `m+1`; initialize
each covariance to identity. Regime banks `rate0,rate1,cue[N,K,J]`; per
channel weight `expert=1/2` with log odds zero. Draw `N` integers via
`factor_rng.integers(5,size=N)` and map into `G=(.0003,.001,.003,.01,.03)` for
particle `hazard`. `entries[N,K]` has first column 1. `used[N]=1`, `z[N]=0`,
particle log weights zero, `started=false`, `t=0`. Place memory and place
background empty; the place key is initially `()`. Slow labels and previous
bits false. Tonic scores `[J,5]`, phasic scores `[J]`, continuation counts
`[J,2]`, and slow counts `[J,2,2]` start at zero. Each target has five vacant
`(source=-1,lag=0)` installed slots. `H=[]`, real prediction cache absent.
The factor source bank has seed `s+103`, capacity 12, self included; maintain
four screening moments per `(target,bank_column,lag)` as in §8.5.

**Common source bank.** Let `w=min(capacity,J)` with self included,
`min(capacity,J−1)` otherwise. Always generate `order=default_rng(seed).permutation(J)`;
`rank` is its inverse; `start=0` with self, 1 without. Complete means
`w=J−start`. Complete rows are ascending eligible indices. Otherwise row
`j` is `order[(rank[j]+start+(0,…,w−1)) mod J]`. Initialize `cursor=w+start`,
`replace=0`. Rotation does nothing on a complete or empty bank. Otherwise
compute `next[j]=order[(rank[j]+cursor) mod J]`, increment cursor modulo `J`.
In each row find the first column in cyclic order from `replace` whose
current source is not pinned; without self require `next[j]≠j`. Replace
only if such a column exists and `next[j]` is not already in that row.
Increment `replace` modulo `w`. Clear the caller's moment arrays at each
changed cell and keep all other cells. Factor rotation pins installed sources;
process rotation pins none.

**Process.** Store the ordered list of singleton groups in ascending original
sensor-coordinate order, each with group seed `s+13*original_coordinate`, HMM
sizes `(2,3)`, and a window. Except as specified in §3, group log weights are
zero, group not started, and no prediction cache. At process level: `tP=0`,
`proposal_number=0`, empty candidate list, empty insertion-ordered rejected
score map, and empty past-outcome list (cap 4). Source bank seed `s+113`,
capacity 12, excluding self. Its four binary moments per lag `0,…,4` start
at zero.

**Reward.** Let `D=5`, `B=16`, `k=min(J,4)`, `C=2^k`, `F=DJ`,
`k_anchor=min(J,4)`. `slots[k]` and `anchor[k_anchor]` are all `−1`;
`known_role[J]=false`. Control bit powers are `2^d`. Observed motion counts
`pooled[J,2,2]`, `by_action[J,A,2,2]`, and predictive log-ratios `[J]`;
`Tcount[A,C,C]`, row totals `[A,C]`, arrival counts `M[C]`, recent rows of
shape `Tcount`; reward bin counts `n[C,B]`, sums `s_r[C,B]`, feature-on
counts `on[C,B,F]`, feature log-ratios `e[C,F]`. Onset counts and amount sums
`[C,F]`, lifetime feature presence `[F]`, current onset occurrence and
past-on totals `[J]`, lifetime update count `total=0`.
Reward mode has separate two scores (recent, archive), `age=0`, `switches=0`,
archive absent. Real replay is a deque of cap `W=2CA`. Context baseline
counts `[J]` and total zero; anchor exposure `[k_anchor]`, on counts and
association scores `[k_anchor,J]`; ordinary context counts and sums
`[C,k_anchor,D,J,2]` plus bin counts and sums with final dimension `B`.
Joint table key `[C,4]` filled with `−1`, margin `edge[C]=0`, and posterior
counts, probabilities and representatives `[C,3,B]` initially zero. Bin
centers `m_k=(k+.5)/B`; precompute `gamma[i]=lgamma(i+1/B)−lgamma(1/B)`
and `dn[i]=lgamma(i+1)` for integers `0,…,W`.

## 3. Process bank

### 3.1 Group members and scores

A group has a list of original coordinates, HMMs in the order given by its
size tuple and a final window member. HMM of size `K_h` has an independent
RNG `default_rng(group_seed+7*K_h)`, count tensor `X[A,K_h,K_h]=0`, initial
emission sum `Y[i,j]=rng.random((K_h,width))[i,j]*.5+.25` (one matrix draw),
and emission mass `M_h[i]=1`. Its filter state is absent until group start;
its buffer of saved transitions has cap 5. Given current counts:

```text
pooled[i,:] ← norm(sum_a X[a,i,:] + .2 + 2 I(i=:))
T_h[a,i,:]  ← (X[a,i,:]+2 pooled[i,:])/(sum X[a,i,:]+2)
E_h[i,j]    ← (Y[i,j]+.5)/(M_h[i]+1)
L_h(y)[i]   ← product_j B(y[j]>.5,E_h[i,j])
```

At group start on `x_g`, set each HMM belief to `norm(L_h(x_g))` (compute
this first likelihood by `exp(sum_j log(max(1e−300,B(...))))`) and set the
window history to the grouped bits of `x_g`. For action `a`, cache
`p=belief @ T_h[a]`; member prediction is `p @ E_h`. For the window, visit
nonempty suffix lengths of its history in increasing order up to 4, keys
`(suffix,a)`. Take the first observed key, then replace it only with longer
keys having count at least 8. Forecast `(count_of_ones+1)/(count+2)` or `.5`
if no key was observed. Cache that forecast. Group prediction is the
weighted sum under normalized exponentiated group log weights. A planning
read of an unstarted group computes what start would produce, without
mutating state. With started groups planning reads the same current
parameters and beliefs without writing any caches.

Real group learning uses the real caches (candidate caches need no readout):

- For an HMM compute `L_h(y_g)` by multiplying probabilities in coordinate
  order. Score is `log(sum(p*L_h))`; new belief `norm(p*L_h)`. Append
  `(old belief,a,y_g)`. When there are 5 buffered transitions, from latest
  to just after oldest send backward messages `beta←norm(T_h[a_i] @
  (L_h(y_i)*beta))`, starting at ones. Using the oldest saved belief `b_o`,
  calculate `xi[i,j] ∝ b_o[i] T_h[a_o,i,j] L_h(y_o)[j] beta[j]`, normalized
  over both indices. Add `xi` to `X[a_o]`; with `g[j]=sum_i xi[i,j]`, add
  `g` to `M_h` and `g[:,None]*y_o[None,:]` to `Y` (**raw** values). Remove
  the oldest logical transition; physical ring length is 5. These updates
  use parameters from before the increment, including for older messages.
- Window score is the sum of Bernoulli log scores on grouped bits, clipping
  its forecast to `[1e−6,1−1e−6]` for scoring only. Increment every key in
  its current history for this action, then append bits of `y_g` and keep 4.
- For member scores `l`, return the group score
  `log(sum(exp(w−max(w)+l)))−log(sum(exp(w−max(w))))` with old log weights
  `w`. Set `w←w+l−max(w+l)`.

### 3.2 Process transitions, screen, and merges

On real prediction prepare incumbent groups in list order and each candidate's
merged, fresh-left and fresh-right groups. On real learn increment `tP`, update
incumbents, then candidates in insertion order. For each candidate accumulate
the merged log score `J_c`, the sum of its two fresh-separate log scores
`F_c`, and the sum of the current incumbent scores of those groups `W_c`.
Set `delta=J_c−logaddexp(F_c,W_c)+log(2)`. Collect all candidates whose
`abs(delta)>bar`; remove and decide them in candidate order. If the original
groups still exist and `delta>bar`, remove the two incumbents and append the
already trained merged group. Otherwise, if both groups still exist, record
the screen score in the rejected map keyed by their ordered coordinate tuples;
if the map already has `max(1,J*bank_width)` entries, evict the first one
before assigning. No other merge or split rule is used.

For each target and source bank column, and each available lag `0,…,4`, use
`source_bit` from `y` (lag 0) or the corresponding element of the past-outcome
list, and use `target_bit` from `y`. Increment moments `n, x, y1, both` by
`1, source_bit, target_bit, source_bit*target_bit`. Append `y[ch]` to the
past-outcome list and keep four. Rotate the source bank every `J` process
updates. The screening statistic is the maximum over lags of

```text
L(both,x−both)+L(y1−both,n−x−y1+both)−L(y1,n−y1)−log(pi)
L(u,v)=lgamma(u+.5)+lgamma(v+.5)−lgamma(u+v+1).
```

Every fifth process update, if there are fewer than six live candidates,
consider directed retained source/target edges between different incumbent
groups that are not in any candidate and whose united width is at most six.
For each group pair in incumbent-list order, keep the largest screen score
strictly above `log(max(1,5*bank_width))+rejected_score` (default 0). Repeatedly
pick the option with lexicographically largest `(score,pair_of_coordinate_tuples)`
and discard all options sharing either group until there are at most six
candidates. Create a fresh merge on sorted union coordinates, seed `s+1000+tP`,
sizes `(2,3,4,6)`; create two fresh separate groups, seeds `+1,+2` relative to
that seed, sizes `(2,3)` for singleton and `(2,3,4,6)` otherwise. Each has a
window and starts on current `y`. Number proposals starting at one; proposal
`i` has `bar=log(1+5*bank_width)+log(i)+log(1+i)+log(4)`. Initialize the three
running scores to zero. Proposals do not contribute to prediction until merged.

## 4. Sensory law

### 4.1 Features and table predictions

For action `a` form one vector per target `j`:

```text
U_j = [h_0[j], h_1[j], h_2[j], onehot_A(a),
       h_l[source] for the five installed slots in slot order].
```

An absent or slow source gives zero. Index each subset's cell by its bits.
Use the following recursion on pooled counts at the selected cells. The
zero-subset prior is `.5`, the singleton prior is the zero-subset prediction,
and a pair's prior is the weighted mean of its two singleton predictions
with weights `exp(eg_i−max(eg_left,eg_right))`. For each cell use
`p_g=(g1+prior)/(g1+g0+1)`. A particle's table member predicts
`p_s=(q1+p_g)/(q1+q0+1)`. Its additive member predicts
`p_add=clip(mu_shared·[U,1]+mu[i,z,j]·[U,1],.01,.99)`.

If there is at least one slow coordinate, mark as current the own lag-0 and
installed lag-0 positions (not action bits). For each pair of current
positions calculate `low=max(active_left*n_left, active_right*n_right)`
from the singleton counts at the selected cells, where `active=current & U`;
`high` is the pooled count at the pair's selected cell. Define `rho` as the
maximum of zero and `(low−high)/(low*(1+high))` over those pairs, using zero
for a zero denominator. If there are no such pairs or no slow bits, `rho=0`.
Compute the mean of all `p_s` and `p_add` with stored weights `Ew`, then
replace it by `(1−rho)*mean+rho*p_add` when `rho>0`. The read visits only
regimes `z<min(kk,used[i]+1)` and fast channels; other outputs are `.5`.
For `z=used[i]<kk`, every particle reads the same pristine table counts,
weights and local coefficients. Evaluate this row once per action and reuse
its output. Particle-specific rates, regime priors and place conditioning
are applied afterwards; occupied regimes are still read separately.

### 4.2 Places, regime prior, and background

Let `P` be the tuple of indices of `slow & b`, `fast=~slow`, and
`quiet=not any(b[fast])`. Set `kk=min(K,max(used)+1)`.

- If factor is not started, `prior[i,0]=1`. Otherwise normalize `entries` in
  the first `kk` columns after adding 1 to column `used[i]` if in range;
  call the result `G_i`. If place `P` is absent, `prior_i=G_i`.
  At a stored place, use `D_i=norm(occupancy_i+G_i)` in that prefix and
  `retain=exp((max(0,t−last_t−1)+1)*log1p(−hazard_i))`. Use
  `D_i*(1−retain)+stored_posterior_i*retain`.
- If some slow channel exists and some fast bit is on, pool cue counts over
  the `kk` regimes within each particle, add `.5` per channel, and normalize
  over channels to `pooled_cue`. For each regime multiply its prior by
  `(sum_active(cue+pooled_cue))/(sum_all(cue)+1)`, where active means
  `fast & b`. Renormalize each row. Else leave the prior unchanged.
- Use unnormalized `w[i,z]=exp(logw[i]−max_i logw[i])*prior[i,z]`.
- Pooled base per particle/channel is `(sum_z rate1+.5)/
  (sum_z(rate1+rate0)+1)` over `kk` regimes. Regime rate is
  `(rate1+pooled_base)/(rate1+rate0+1)`. If there are no slow bits, set
  `B0=0`, and the local prediction is the read of §4.1, except for unused
  regimes, where it is `.6*read+.4*rate`. If slow bits exist, let `B0=rate`
  or, when place background exists, `(place_ones+rate)/(1+place_count)`.
  On quiet input the local prediction is `B0` and no feature read is needed.
  Otherwise `local=B0+(1−B0)*read*fresh_factor` with `fresh_factor=.6`
  when `z≥used[i]`, 1 otherwise. All base rates here are pre-update.

### 4.3 Component probabilities and the quiet clock

Compute `p_proc[a,j]` by §3.1 and set
`p[a,i,z,j]=expert[j]*local[a,i,z,j]+(1−expert[j])*p_proc[a,j]`.
For each slow `j`, its independent marginal is
`p_slow[j]=(slow_count[j,b_j,1]+.5)/(sum_k slow_count[j,b_j,k]+1)`.
For the following joint law, flatten `i,z` in row-major order and normalize
`w`. Conditional on a component `q`, the fast law off quiet input is the
product of its Bernoulli probabilities. For quiet input with some fast
channels define

```text
idle_q=product_{j fast}(1−p[q,j]); idle=sum_q w_q idle_q
P(all fast zero)=idle
P(fast pattern u not all zero)=(1−idle)*sum_q w_q P_q(u)/(1−idle_q).
```

The quiet marginal multiplier for fast predictions is
`−expm1(min(0,log_idle))/max(−expm1(log_idle_q),1e−300)`, where
`log_idle_q=sum_fast log1p(−p[q,j])` and
`log_idle=LSE(log(unnormalized_w)+log_idle_q)−log(sum(unnormalized_w))`.
When not quiet or fast is empty, the multiplier is 1. The slow marginals
replace their corresponding coordinates after mixture averaging.

### 4.4 Restricted marginals and draw operation

For a requested set of at most three coordinates in ascending order,
enumerate bit patterns in increasing binary-code order (first coordinate
is the least significant). For the retained fast bits `u`, take each
component's product on those bits. On quiet input with retained fast bits,
if `u` is all zero, multiply this product by the probability of at least one
success among the omitted fast bits, `−expm1(sum_omitted log1p(−p_q))`.
Divide each component's result by `1−idle_q` (zero if this mass is zero),
mix with weights `w_q`, multiply by `1−idle`, and add `idle` only to the
all-zero fast pattern. Without a quiet clock or without retained fast bits,
just mix the products. Multiply slow-coordinate likelihoods and normalize
the resulting pattern masses. Other coordinates of each returned vector
are zero.

For a draw use each row `u` of `U` as follows. Column 0 selects a
component via `searchsorted(cumulative_weights,u[0]*total,side='right')`,
clamped to the last component. Columns `3:3+J` sample its fast bits and
the slow marginals using `<`. If quiet with some fast bits, use column 1
for the shared idle event (`u[1]<idle`); set all fast bits to zero on idle.
Otherwise use column 2 to choose the first success among the fast bits:
form the cumulative distribution `−expm1(cumsum(log1p(−p_q)))`, lookup
`u[2]*last_cdf` with `side='right'` clamped to the last fast position.
Clear positions before that success, set it, and keep the original draws
for later positions. Return the five draws in row order with mass `1/5`.
Column `J+3` is generated but not used.

### 4.5 Real prediction preparation

Use key `(bits(x),a)`. If it already matches the cache, reuse the cached
prediction and do not prepare the process bank twice. Otherwise use §§4.1–4.3
with current state and *real* group preparation in §3. Cache the selected
keys, pre-update `p_g`, feature vectors and deficit, process marginals,
prior, unnormalized weights, local component probabilities, background, and
mixed prediction. Prediction is `.5` at `v`. Only real prediction fills
this cache and the group learning caches.

## 5. Reward learning

### 5.1 Features, control code and real storage order

For `l=0,…,4` let feature `X[l,j]=h_l[j]` and onset
`O[l,j]=h_l[j]*(1−h_{l+1}[j])`. Flatten in row-major `(lag,channel)` order.
Real reward `r` must be finite and in `[0,1]`; assign bin
`k_r=min(15,int(16*r))`. For a vector `b`, the control code is
`c(b)=sum_d 2^d I(slot[d]≥0)*b[max(0,slot[d])]`. Valid codes have no bit on
an absent slot. For a control row, define

```text
base=norm((M+.5)*valid)
T[a,i,:]=(Tcount[a,i,:]+base)/(row_total[a,i]+1).
```

At entry to reward learning, retain the old slot vector and switch count.
Increment lifetime `total`, presence by `X`, `onset_total` by `O[0]`, and
`past_on` by `b`. Append `(b,a,r,b',X,O)` to the capped replay. For each
channel calculate the pre-update Bernoulli log-score ratio of the
`(a,b[j])` row to the pooled `b[j]` row for bit `b'[j]`, using Jeffreys
smoothing. Add to that channel's motion score, then increment both rows.
Define `p[j,a,u]=(by_action[j,a,u,1]+.5)/(sum by_action[j,a,u,:]+1)`.
An eligible mark has all of:

```text
max_u(max_a p−min_a p)>.5;
max_a p[j,a,1]−min_a p[j,a,1]>.5;
motion_log_ratio[j]>log(1+J).
```

OR eligibility with `known_role`. Remove ineligible installed marks, then
fill empty slots in ascending order by the highest-score eligible unused
mark (first maximum); stop when none remains.

If slots changed, discard archive, zero recent rows, both mode scores,
and `age`, zero all active mode banks listed below, and record each replay
row in oldest-first order with mode switching disabled. The limit and
contents of the real replay do not change. Otherwise record just its newest
row with mode switching enabled. Slot changes do not zero the motion
screen or lifetime presence/onset/past-on counts.

### 5.2 Recording and switching modes

Mode banks are exactly `Tcount`, row totals, `M`, `n`, `s_r`, `on`, `e`,
onset counts and onset amount sums. For row
`(b,a,r,b',X,O)`, let `i=c(b)`, `c=c(b')`, evaluate the *same current* base
above, and compute pre-record probabilities of `c` under the current row,
the recent row, and the archived row (use current if there is no archive).
If not replaying, row_total is positive, and some slot is live, update both
scores as `score←max(0,score+log(P_alt/P_current))`. Take first argmax. If
its score strictly exceeds `log(1+age)+log(1+#valid_codes)`, save the old
active banks; restore the archived banks if the winning alternative is
archive and it exists, otherwise zero the active banks. Replace the single
archive with the saved banks, mark all current slots in `known_role`, clear
the replay, zero both scores and `age`, increment `switches`, and append the
current row as the first row of replay. Do **not** zero the recent rows on
this kind of switch. No action permutation is tested.

Next multiply just the recent row `[a,i,:]` by `.5`, add 1 at `c`, and
increment `age`. Add 1 to `[a,i,c]` and its row total, then update the reward
law. For each feature calculate pre-update proportions

```text
pooled_on=(sum_bins on[c,:,f]+.5)/(sum_bins n[c,:]+1)
bin_on=(on[c,k_r,f]+.5)/(n[c,k_r]+1)
e[c,f] += log(B(X_f,bin_on)/B(X_f,pooled_on)).
```

Finally increment the bin count and `M[c]` by 1, bin amount by `r`, bin
feature-on counts by `X`, onset counts by `O`, and onset amount sums by
`r*O` at arrival `c`.

### 5.3 Anchor and context update

After §5.2 set `intensity=(onset_total)/(.5+past_on)` and
`strength=intensity*(1−(past_on+.5)/(1+total))*sqrt(onset_total)`, zero at
controlled coordinates. Choose the desired set via
`argpartition(-strength,k_anchor−1)[:k_anchor]`, keeping its returned order
and dropping zero strengths. Replace undesired anchor slots in ascending
slot order with previously uninstalled desired indices in that order, or
`−1` if exhausted. At changed slots zero the corresponding exposure,
association and anchor-conditional on counts. For each anchor currently
on in `b`, add log-ratios of `(.5+pair_on)/(1+anchor_exposure)` versus
`(.5+context_x)/(1+context_t)` for the current vector `b`; then increment
its exposure and add `b` to `pair_on`. After all anchors add `b` to the
global context vector and increment `context_t`.

If anchors, control slots, or switch count changed, zero all ordinary
context counts and sums (including bin tables) and replay the current window
in order. Otherwise use only the last replay row. For a replay row at
arrival `c` and each current anchor slot `d` with mark `j`, for every
`l` with `O[l,j]=1`, update the entry `[c,d,l,m,X[l,m]]` for **each** sensor
`m`, including `m=j`: count `+1`, sum `+r`, and the same count and sum at
bin `k_r` in the bin tables. No re-estimation of the anchor screen occurs
inside this replay.

### 5.4 Joint selector

After §5.3 reset all joint keys to `−1` and margins to zero. Stop if no
control slot is live or the replay is empty. Unary list: all noncontrolled
features in ascending `(lag,channel)` order. Compound list: in ascending
anchor-slot order, for each installed noncontrol anchor `j`, consider
noncontrol moderators `m≠j` in ascending order with
`(pair_on[d,m]+.5)/(anchor_exposure[d]+1) > (context_x[m]+.5)/(context_t+1)`
and association `>log(1+J)`. For each moderator append lags `0,…,4` in order.
Stop if there are no compounds.

In each arrival `c`, pool every replay reward into its bin; require at
least three rows. Recode each row using the current slots. Let
`ell(n)=sum_k gamma[n_k]−dn[sum_k n_k]` for bin counts in any group.
For a unary candidate use two groups by the feature bit; skip if either is
empty. Compute `ell(on)+ell(off)−ell(pool)−log(max(1,#unaries))`. Set
threshold to the maximum of 0 and all these scores. For a compound use
three groups: `X[l,m]` when anchor onset `O[l,j]` is true, otherwise group 2.
Skip any compound with an empty group. Score it by the sum across the three
`ell` values minus `ell(pool)+log(max(1,#compounds))`. Choose the first
compound reaching the largest score *strictly above* threshold. If none,
leave this arrival unselected. At selection store `(j,m,l,d)`, margin
`best−threshold`, and for each group store the bin law of §6.1 from that
window. Old posterior tables for unselected arrivals are unobservable and
need not be cleared.

## 6. Reward readout

### 6.1 Bin law and feature likelihood

For counts `n_k` and sums `s_k` use

```text
alpha_k = n_k+1/16
P_k     = alpha_k/sum(alpha)
U_k     = (s_k+m_k/16)/alpha_k.
```

For each arrival select the highest `e[c,f]` among noncontrol features
(mask others with `−inf`). Use one candidate via `argpartition(-score,0)`;
credit it only if its score is greater than `log(1+F)`. If joint margin is
positive, replace that candidate by `f=lJ+j` from the joint key and set it
credited. For every credited feature, use conditional on-probability
`(on[c,k,f]+.5)/(n[c,k]+1)`. For a queried `X`, define

```text
lik[c,k]  = B(X[f],conditional_on[c,k]) if credited else 1
atom_mass = (M[c]+1)*P[c,:]*lik[c,:]
Z[c]      = sum(atom_mass)
prob[c,:] = atom_mass/Z[c]
mean[c]   = sum(prob[c,:]*U[c,:]).
```

`P,U` here are the arrival's active bin law. Preserve `lik,Z` from this
step for continuation corrections even when overriding the returned
mean/probabilities afterwards. A readout has no learning side effects.

### 6.2 Signed rules and group IDs

For each installed anchor `j` in slot `d`, form eligible `m≠j` with
conditional on-rate above baseline and association `>log(1+J)`, excluding
controlled channels. For each lag `l`, only arrivals whose credited unary
feature is `lJ+j` and which have no joint selection are eligible for the
ordinary rule. For each eligible moderator let `n_0,n_1` be ordinary
context counts at `c,d,l,m`, and `u_0,u_1` their sample means. Both counts
must be positive. With `n=n_0+n_1` compute

```text
raw_gain = n_0*n_1/max(1,n)*(u_0−u_1)^2
excess = raw_gain − .25*log(1+n) − .25*log(#eligible_moderators).
```

Select the first argmax among positive excess at each `c,d,l`. Store rules
in anchor-slot order, then lag order, then ascending distinct winning
moderator, with an arrival-support mask and the raw gains. Treat the global
signed-rule list as nonempty if there is any ordinary rule or any positive
joint margin. For an arbitrary query `(X,O)` initially set group ID and
group mass to 0. At a nonjoint arrival, among its supported rules with
`O[l,j]=1`, choose the largest raw gain, retaining the first on a tie. Let
`g=X[l,m]`; replace its law by §6.1 applied to the ordinary context table
for this branch, set ID `1+2*(d*5+l)+g` and mass `sum alpha`. At a joint
arrival override the law for *every* query: `g=X[l,m]` if `O[l,j]` else 2;
use its stored joint law and set ID `1+2*k_anchor*5+3*(d*5+l)+g`, mass
`sum alpha`. Recompute the mean from the returned probabilities and amounts.
If no rule applies, use the unary/default law unchanged. For a query with
no signed-rule list, group-matching logic is omitted.

### 6.3 Upper offers

For each onset feature at an arrival, `n_o,s_o` are active onset count and
amount sum, `m=(s_o+.5)/(1+n_o)`,
`intensity[l,j]=onset_total[j]/(.5+past_on[j])`, and
`prevalence=(presence+.5)/(1+total)`. Its offer is

```text
bound=max(0,m+(1−m)*intensity*(1−prevalence)/sqrt(1+n_o)).
```

Zero at controlled features and arrivals with a credited feature. Multiply
by the query onset bits and take the maximum over features.

For the paired offer, loop anchor slots in ascending order. Its candidate
moderators satisfy the association/on-rate screen of §5.4 and also
`conditional_on < 1−.5/(1+anchor_exposure)`. Use the first maximum
association; skip if its association does not exceed `log(1+J)`. For each
queried active onset of this anchor (skip lag 0 if `shift=true`), set
`g=X[l,m]`, `p=conditional_on[m]`, `p0=baseline[m]`, and
`rarity=(1−baseline[j])*onset_total[j]/(.5+past_on[j])`. Use the ordinary
context counts and sums for both branches:

```text
prior=(s_0+s_1+.5)/(1+n_0+n_1)
mu=(s_g+prior)/(1+n_g)
factor=rarity*(p−p0)/max(1e−12, p if g=0 else 1−p0)
pair_offer=mu+factor*(1−mu)/sqrt(1+n_g).
```

The offer is zero unless the arrival credits `lJ+j`. Take the maximum
starting from zero across anchors/onsets. At the caller, mask the paired
offer wherever group ID is nonzero.

### 6.4 Soft root and repeated decision option

For each installed eligible-at-query anchor `j` (current onset `O[0,j]`
true and exposure at least two), use the moderator screening and argmax
of §6.3. If its association clears the threshold, yield the lag-0 ordinary
context histogram at `m,g=X[0,m]` for each arrival, flag `paired=true`, and
frequency `((onset_total[j]+.5)/(total+1)) * (p if g else 1−p)`. Otherwise
yield the lag-0 histogram at `m=j,g=1`, flag false, and only the first
frequency factor. Process yielded candidates in anchor order.

For subset histogram `n1` and active whole `n0`, define

```text
L(n) = sum_k (lgamma(n_k+1/16)−lgamma(1/16))−lgamma(1+sum n)
odds = −log(k_anchor)+L(n1)+L(max(0,n0−n1))−L(n0)
lambda = 1/(1+exp(−clip(odds,−30,30))).
```

For **root refinement**, use paired candidates only, with positive subset
mass and no credited feature. Among them choose by greatest **unclipped**
odds (first ties). From the unmodified root law `p,u` and the subset law
`p1,u1` of §6.1 return

```text
p_new=(1−lambda)*p+lambda*p1
u_new=((1−lambda)*p*u+lambda*p1*u1)/p_new.
```

Replace the mean by its weighted value and mark the arrival as refined.
Retain the existing group ID. All others keep their original laws.

For the **repeated option**, define `age1=1+mode_age` and
`H=sum_{i=1..2W} age1/(age1+i)`. Given fixed root values `q` (§7), and for each
candidate and arrival with `M[c]>0`, if the arrival credits a feature,
require `paired=true` and credited feature equal to `j` at lag 0. Let
`reach[a]=T[a,origin,c]`; skip if `max(reach)−min(reach)<1e−12`. Use the
whole law `P0,U0` and subset law `P1,U1`, both from §6.1 with corresponding
real bin sums, and the same odds above. Set `m0=(sum whole_amounts+.5)/(1+sum n0)`
(equivalently implement as `sum(whole_amounts+m_k/16)/(1+sum n0)`),
`m1=sum(P1*U1)`, and calculate

```text
P[k]=(1−lambda)*P0[k]+lambda*P1[k]
m_after[k]=((1+sum n1)*m1+U1[k])/(2+sum n1)
G[k,a]=P[k]*q[a]+lambda*P1[k]*(m_after[k]−m0)*reach[a]
v=max(0,sum_k max_a G[k,a]−max_a sum_k G[k,a])
option[a]←max(option[a],(reach[a]−min(reach))*H*frequency*v).
```

Start `option=0`. There is no additional refinement of `q` inside this scan.

## 7. Action values

Given `x,U`, compute `b=bits(x)` and the root `X,O`.

**Constant context, action-dependent reward.** Consider this partition only
if `A≥2`, no controlled slot is live, and `O` is all zero. Each replay row
must have arrival bits equal to `b`, features equal to the root `X`, and
all-zero onsets. There must be at least two rows for each action. For each
action `a` count and sum its replay rewards by bin. Let `n_a` be its bin
counts. Choose the action partition only if
`sum_a ell(n_a) > ell(sum_a n_a)`, with `ell` as in §5.4. Both candidates
score exactly the same rows.

When the action partition wins, use §6.1 on each action's bins to obtain
`P_a[k]`, `U_a[k]` and mean `m_a=sum_k P_a[k]*U_a[k]`. Set
`N_a=sum_k n_a[k]`. Return early from this query with

```text
next_a[k] = m_a + (U_a[k]−m_a)/(2+N_a)
Q[a] = m_a + sum_k P_a[k]*clip(max(next_a[k], max_{j≠a} m_j),0,1).
```

This is the one-context, two-step case: the atom updates only its action
partition. There is no extra exploration bonus or hypothetical transition
update. Real prediction, learning, and RNG calls in §1 do not change.

Otherwise let `origin=c(b)` and `T` be §5.1. Use §§6.1–6.2 and then root
refinement from §6.4 to obtain root mean `m0[c]`, probabilities `p0[c,k]`,
representatives `u0[c,k]`, group
IDs `g0[c]`, and refined-arrival mask. Optimistic immediate `I[c]` is the
maximum of `m0[c]`, §6.3 single onset offer, and its paired offer at group
ID 0. For refined arrivals force `I[c]=m0[c]`. Even where `I` differs, use
`p0,u0` for first reward atoms. Discount is exactly 1.

Let `states` be ascending codes `c` reached with `T[a,origin,c]>0` under at
least one action; let `witnessed[c]=(M[c]>0)`. Relevant sensory coordinates
are the sorted unique union of lag-0 credited features and lag-0 moderators
of any signed rule. With at most three such coordinates, the ordered branch
support is the same for every first action. The implementation builds the
conditional continuation tables in steps 1–4 once per query, up through
reduction over reward bins. The branch masses and arrival row stay specific
to each action, and their contractions retain the order below. With sampled
branches, build the continuation tables separately. The following equations
describe the result for each action:

1. If no relevant coordinate, use one zero sensory vector and mass 1. If
   one to three, use §4.4 restricted patterns and masses for that action.
   Otherwise use the five draws and their equal masses (§4.4). For each
   sensory branch `beta` in order and `c` in `states` order, overwrite its
   live controlled bits with the code `c`, giving `b1`. Compute the
   continuation features and onsets using `(b1,H+[b])`.
2. Run §§6.1–6.2 for those features at every second arrival `d`. Denote the
   result `m2[d]`, `g2[d]`, group mass `N2[d]`, and preserve the likelihood
   `L2[d,k]`, `Z2[d]` from §6.1. There is no second soft refinement. Derive
   `cont[d]` as the maximum of `m2[d]`, the single offer with all lag-0
   onsets masked, and the paired offer with `shift=true` only at `g2[d]=0`.
3. For each bin `k` compute a change to mean of arrival `c` only:

```text
diff[k] = u0[c,k]−m2[c]
Delta[k] = (L2[c,k]/(Z2[c]+L2[c,k]))*diff[k]
if g2[c]>0:   Delta[k] = diff[k]/(1+N2[c])
if signed rules exist and g2[c] != g0[c]: Delta[k]=0
```

4. Compute for each second action `j`:

```text
v_fixed[j] = T[j,c,:] @ m2
v_opt[j]   = T[j,c,:] @ cont
peak       = max(cont[d] at witnessed d), or max(cont) if none witnessed
v_opt[j]  += (peak−v_opt[j])/(2+row_total[j,c])
branch_value(kind,k) = clip(max_j (v_kind[j]+T[j,c,c]*Delta[k]),0,1).
```

   For each `c`, `future_kind[c]` is the sum over `beta,k` of
   `mass[beta]*p0[c,k]*branch_value(kind,k)`. Reduce bins first and then
   residual branches. Use fixed values only when a control slot is live;
   otherwise omit them.
5. For this root action, `Zopt[c]=I[c]+future_opt[c]`. Set
   `Qopt=T[a,origin,states] @ Zopt`. Add `(max(Zopt at witnessed states)−Qopt)
   /(2+row_total[a,origin])`, using all enumerated states when none is
   witnessed. If computing fixed values, use
   `Qfixed=T[a,origin,states] @ (m0[states]+future_fixed)` with no root pull.

After all actions, return `Qopt` if no control slot is live; otherwise
return elementwise `max(Qopt,Qfixed+option)` using `option` from §6.4 with
`q=Qfixed`. In the fixed path use **the same** residual branches and atom
corrections as in the optimistic path. No model state or RNG is updated
inside this calculation.

## 8. Real sensory update

### 8.1 Roles, process update, and place boundary

With `b=bits(x)`, for each channel whose saved previous origin bit was 1,
update `tonic[j,h] += log(B(b[j],1−G[h]))` on the grid, and
`phasic[j] += log((on_count[j,b[j]]+1)/(sum on_count[j,:]+2))`; then increment
that count. If there were any previous ones, compute for all channels

```text
a[j,h] = tonic[j,h]−phasic[j]−log(J)
basis[j,h] = logaddexp(0,a[j,h])
rest[j,h]  = sum_i basis[i,h]−basis[j,h]
slow_new[j] = (LSE_h(rest[j,h]+a[j,h])−LSE_h(rest[j,h]) > 0).
```

Otherwise keep the old labels. Save current `b` as previous origin. With
new labels obtain `P=tuple(indices where b & slow_new)`. Clear each installed
parent whose source just became slow (leave its lag field), performing the
slot reset of §8.5. A real sensory cache is reusable iff key `(b,a)` matches
and labels did not change. If not reusable, prepare only the real process
bank again on `(x,a)` to obtain its old marginal. Now learn the process
bank from the real `y` (§3). Increment the two-row slow counts for **all**
channels at `(j,b[j],b'[j])`. Append `b` to `H`, retain the last 5, and call
the truncated list excluding this newly appended `b` `H_before`. Clear
the real sensory cache and store `P` as current place. If the new slow-bit
signature of `b'` differs from `P`, increment `t` and return **without**
executing §§8.2–8.5 or drawing from the factor RNG.

### 8.2 Mixed forecast used by screening, and regime filtering

Take cached values from §4.5 if reusable; otherwise compute §§4.1–4.3
using `H_before`, current factor state, and the pre-learning process
marginal. In either case retain the mixed marginal `q_pred` **before**
overwriting slow channels; for a quiet read use the multiplier of §4.3
on the component weights for this contraction over all channels. Use
cached pre-update `B0`, `local`, `prior` and `w` when possible.

If no slow channel exists or the input is nonquiet, for each fast channel
compute `p_local=sum_{i,z} w*local/sum(w)` (without the quiet multiplier)
and add to its expert log odds `log(max(1e−12,B(b',p_local))) −
log(max(1e−12,B(b',p_proc)))`. Set its weight to
`exp(−logaddexp(0,−expert_log_odds))`. No other expert weights change.

For filtering, form the §4.3 quiet multiplier using **local**, not mixed,
component probabilities. If that multiplier exists and all fast `b'` bits
are zero, use `filtered=prior` and do not change `logw`. Otherwise set

```text
log_joint[i,z] = log(prior[i,z]) + sum_{j fast} log(B(b'[j],local[i,z,j]))
                 + log(local_quiet_multiplier[i,z])  # term only if it exists
filtered[i,:] = softmax(log_joint[i,:])
logw[i] += LSE_z(log_joint[i,z]).
```

Draw exactly `N` uniforms from the factor RNG. Per row set
`z_new=min(number of prefix cumulative probabilities strictly less than U_i,kk−1)`.
Increment `entries[i,z_new]` iff `z_new≠z[i]` or `z_new≥used[i]`;
set `used[i]=max(used[i],z_new+1)`, `z=z_new`, `started=true`.

### 8.3 Place and table increments

If place `P` is new, evict the least-recently-used place and its background
when already at capacity 10; create a record with no posterior, zero
occupancy, and last time `t`. Move `P` to the most-recent position. Mask
`filtered` beyond updated `used`, normalize each row and pad to `K` with
zero; call it `posterior`. If any old or new fast bit is on or the place
has no stored posterior, add this posterior to occupancy and store it
with mark time `t`. Otherwise do not replace mark time or posterior.

If slow bits exist and input is nonquiet add the active fast bit vector to
`cue[i,z_new,:]`. On quiet input or if there are no slow bits increment
`rate0/rate1` at `i,z_new` for each fast outcome. When slow bits exist and
input is quiet, increment only the fast outcome counts in `P`'s background
and skip the member and pooled-table updates below. Otherwise:

1. For each particle and fast channel use
   `credit = max(0,local[i,z_new,j]−B0[i,z_new,j])/max(1e−12,local[i,z_new,j])`
   for `b'[j]=1`, otherwise 1. Set transformed additive probability
   `P_add=B0+(1−B0)*p_add`, score its observed Bernoulli likelihood `L_add`,
   and add `log L_add` to the additive member's log score.
2. For each table member at the pre-update selected cell, compute
   `P_table=clip(B0+(1−B0)*(q1+p_g)/(q1+q0+1),1e−6,1−1e−6)`.
   With `L_table=B(b',P_table)`, `L=(1−rho)*L_table+rho*L_add`, add `log L`
   to its log score. Increment the observed bit count by `credit` if
   `rho=0` or by `credit*(1−rho)*L_table/max(1e−12,L)` otherwise. Update
   cached weights `Ew=exp(ev−max(ev))` in each sampled particle/channel row.
3. Apply the Gaussian update (§8.4) with the same credits and pre-update
   shared centers. Recompute the pooled predictions at the selected
   cells from the old pooled counts/log scores and clip to
   `[1e−6,1−1e−6]`. Use `Bbar=sum_{i,z} w*B0/sum(w)` with the **prediction**
   weights, and `w_post=softmax(logw)` with the **updated** log weights.
   Add `log B(b',Bbar+(1−Bbar)*p_g)` to each pooled log score on fast bits.
   Add `sum_i w_post[i]*credit[i,j]` to the chosen pooled count in each
   selected cell on fast channels only.

### 8.4 Gaussian update and particle resampling

For each fast channel, use `vfeat=[U,1]`, and target
`target=b'[j]−cached_shared_center[j]`. For each particle and then the
shared array, use that particle's credit or (for shared) the ordinary
mean of the particle credits. If credit is zero, leave the row unchanged.
Calculate `sv=Cov @ vfeat` and `var=.25/credit+vfeat @ sv` using the
nonzero feature positions for `sv`; compute residual `target−mu @ vfeat`
for particle rows and just `target` for shared. Update
`mu += (sv/var)*residual`, `Cov -= outer(sv/var,sv)`. Do not move the shared
update before the particle updates. Table counters and their likelihoods
are independent of this operation's in-place covariance writes.

After §8.3 (also when only background updated), normalize particle
log weights. If `1/sum(w_post^2) < N/2`, draw one more factor uniform `u`.
For destination particle `d`, choose first ancestor whose cumulative
normalized weight is at least `(u+d)/N`, falling back to the last. Gather
simultaneously from old particles: the factor table and Gaussian banks,
`entries`, `cue`, `rate0/1`, all place posteriors and occupancy arrays,
and hazard, regime and live-prefix vectors. Resampling leaves shared
banks untouched; unused rows outside both source and destination prefixes
already equal their initial values. Set all particle log weights to zero.
The factor RNG is never called for resampling if the inequality fails.

### 8.5 Screening and slot resets

Use residual `e[j]=b'[j]−q_pred[j]`. For each lag `l≤len(H_before)` in
`0,…,4` and each retained source, take
`u=h_l(b,H_before)[source]`. Increment its moments
`(n,x,r,xe)` by `(1,u,e[j],u*e[j])`; set its score to
`(xe−x*r/n)^2 / (1+x*(1−x/n))`. Increment `t` before slot selection.

For each target find the first weakest installed slot by score, using
`−inf` for absent slots or sources no longer in the bank. Scan bank
columns then lags, choose the first highest scoring available pair;
exclude slow sources, own source with lag `0,1,2`, and any source
already installed at any lag on that target. Replace the weakest slot
only if the new score exceeds `.25` and exceeds the old score by `.25`
(strict in both comparisons). Determine all choices before applying
resets. Rotate unpinned bank columns only when incomplete and `t mod J=0`;
rotation follows §2 and zeros all score/moment entries for changed columns.

**Reset a feature position** in a target: for every subset containing the
position, zero all its pooled cells and copy the log score from the subset
without it. Do the same with particle cells and log scores for **all**
particles up to the old global `max(used)` regime prefix. For that feature
zero the shared and corresponding particle coefficient and covariance row
and column, setting the variance to 1. Recompute all cached member
weights for the target across the full `K`, preserving other coordinates.
Removing a newly slow source uses exactly this reset before prediction
recomputation. No other parameter of that slot's target is reset.

## 9. Numerical conformance

Use row-major ordering for the tuples and bit-pattern enumerations above.
The reference uses IEEE binary64 unless specified otherwise; feature
and mask computations use exact indices or booleans. `q0,q1,ev,Ew` are
binary32, and so are `p_g` at particle-member reads/updates, the clipped
additive prediction there, and `B0`, `rho`, and credit inside the particle
count/score update. The Gaussian update uses binary64 credits and centers.
In the table kernels use binary32 constants for bounds `1e−6`, `1−1e−6`,
floor `1e−12`, and unit table prior; round stored counts and scores to
binary32 on each update. Pooled counts/scores and the pooled update in
§8.3 use binary64. In the HMM step multiply the likelihood in ascending
channel order; use `log` once after the state-weighted reduction. In the
Gaussian update accumulate `sv` in ascending nonzero position order,
then the variance and residual in ascending position order. In the
factor member read start numerator and denominator with the additive
member, then accumulate table members in subset order. In the joint
selector initialize a compound score to `−ell(pool)−cost`, then add the
three `ell` values in order `0,1,2`.

For the categorical operator used by planning, form `Tcount+base` first
and divide each row by its computed sum; the mode-witness row division in
§5.2 uses `1+sum(count_row)` instead. Likewise use the prefix cumulative
sum and the strict comparisons literally, not an approximate tie tolerance.
Use `log1p` and `expm1` where specified, not subtraction-based replacements
in the quiet or regime-survival paths. The reference uses
`argpartition`, permutation and RNG operations of NumPy 2.4.6 (default
PCG64). Loop formulas define their mathematical results; reassociation,
libm implementation and BLAS reductions can change floating-point results
at the last bit. Action equivalence requires exact action agreement on a
conformance tape; a nonzero tolerance on `Q` does not establish action
agreement near a tie. For a binary regression against the vectors below,
reference stack is CPython 3.11.2, NumPy 2.4.6, Numba 0.68.1, llvmlite 0.50.0,
Linux x86-64/glibc 2.36, one OpenBLAS and one Numba thread. No fast-math
assumptions enter the updates.

### 9.1 Closed-loop check vector (no external input data)

`n=6, A=3, v=5`, initially `pos=0` and `x=[1,0,0,0,0,.5]`. For each seed
start a new controller, then for `t=0,…,95`:

```text
a ← step(x)
pos ← (pos+a) mod 3
cue ← I((t+1) mod 7 ∈ {0,1})
modulator ← I((t+1) mod 11 < 3)
r ← .5 if pos=1, else .45 if pos=2,
     else 1 if x[3]=1 and x[4]=0, else 0
y ← [I(pos=0),I(pos=1),I(pos=2),cue,modulator,r]
learn(y); x ← y
```

Expected actions, concatenated with no separators, and SHA-256 of the `Q`
array written as little-endian binary64 in step-major/action-minor order:

```text
seed 7
121221000001000210000111201002101000120000120000010100002010000201000020100002010000201000020100
deb284871124615b32327e872f1936e53c85a70116c05f43292df2a3b93cf90c

seed 29
202000022112000022000202102220200202002200201000021121021102200212220110100002020110222000021000
415003393206959c10b0b183b72112eb9bce4aee1c3c6b9eac43685a3d22b85b
```

Both seeds have `Q=(1,1,1)` at step 0. The `Q` vector at step 95 in C99
hexadecimal floating-point notation:

```text
seed 7   0x1.0e258c033a974p+0  0x1.ff95749c171e2p-1  0x1.48bc95f1e6b49p-1
seed 29  0x1.2721eb5e0c121p+0  0x1.263697cee1a6ap+0  0x1.16e0dd6ac8372p+0
```

After the last learning step: max `used` is 3 in both cases; installed
controlled slots are `[0,1,2,−1]` and `[1,0,2,3]`, respectively; there are
no joint selections or mode switches at the final checkpoint.

### 9.2 Joint-score check vector

For §5.4, retain twelve rewards at one arrival. Supply one compound with
four observations in each group, with joint group histograms supported in
three distinct bins. Supply two unaries, each isolating one pure group.
The pooled log marginal is `−22.59365555163929`, the best penalized unary
score is `5.5114105820087484`, the compound score is `10.453053004618052`.
The group with four rewards `.9` has bin-14 probability `.8125`, bin-14
representative `.9000961538461538`, and mean `.82`. Removing all rows of
one compound group makes the compound ineligible. This checks §5.4 and
§6.1 independently of the sensory/actuator screens. The closed-loop check
vectors do not cover every state, admission, or mode-recurrence branch.
