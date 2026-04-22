type AgentOption = { value: string; label: string };

type BatchAgent = {
  agent: string;
  episodes: number;
  accuracy_sum: number;
  reward_sum: number;
  steps_sum: number;
  success_sum: number;
  failure_sum: number;
  extra_sums?: Record<string, number>;
  trial_curve_sums?: Record<string, number[]>;
};

type ExtraMetricSpec = {
  key: string;
  label: string;
  short_label?: string;
  format?: 'number' | 'percent';
  decimals?: number;
  denominator?: 'episodes' | 'success';
};

type TrialCurveSpec = {
  key: string;
  label: string;
  short_label?: string;
};

type ExperimentDefaults = {
  episodesPerAgent: number;
  batchSize: number;
  stepCap: number;
};

type BatchResult = {
  scenario: string;
  scenario_name: string;
  accuracy_label: string;
  reward_label: string;
  step_cap: number;
  extra_metrics?: ExtraMetricSpec[];
  trial_curves?: TrialCurveSpec[];
  agents: BatchAgent[];
};

type AgentTotals = {
  episodes: number;
  accuracySum: number;
  rewardSum: number;
  stepsSum: number;
  successSum: number;
  failureSum: number;
  extraSums: Record<string, number>;
  accuracyHistory: number[];
  rewardHistory: number[];
  extraHistories: Record<string, number[]>;
  trialCurveSums: Record<string, number[]>;
};

const AGENT_COLORS: Record<string, string> = {
  combined: '#a78bfa',
  active_learning: '#f59e0b',
  active_inference: '#06b6d4',
  greedy: '#22c55e',
  random: '#ef4444',
};

export class ExperimentDashboard {
  private viewportRoot: HTMLDivElement | null = null;
  private chartGridEl: HTMLDivElement | null = null;
  private tableEl: HTMLDivElement | null = null;
  private statusEls: HTMLDivElement[] = [];
  private accuracyLabel = 'Accuracy';
  private rewardLabel = 'Avg Reward';
  private extraMetrics: ExtraMetricSpec[] = [];
  private trialCurves: TrialCurveSpec[] = [];
  private trialCurveLength = 0;
  private totals = new Map<string, AgentTotals>();

  constructor(
    private readonly scenarioId: string,
    private readonly scenarioName: string,
    private readonly agents: AgentOption[],
  ) {
    this.resetResults();
  }

  init(viewport: HTMLElement): void {
    viewport.innerHTML = `
      <div class="exp-shell">
        <div class="exp-header">
          <div>
            <div class="exp-kicker">Experimenter</div>
            <div class="exp-title">${this.scenarioName}</div>
          </div>
          <div class="exp-status" id="exp-status-viewport">Ready to benchmark all agents.</div>
        </div>
        <div class="exp-grid" id="exp-chart-grid"></div>
      </div>
    `;
    this.viewportRoot = viewport.querySelector('.exp-shell');
    this.chartGridEl = viewport.querySelector('#exp-chart-grid');
    const viewportStatus = viewport.querySelector('#exp-status-viewport') as HTMLDivElement | null;
    this.statusEls = viewportStatus ? [viewportStatus] : [];
    this.render();
  }

  dispose(): void {
    this.viewportRoot = null;
    this.chartGridEl = null;
    this.tableEl = null;
    this.statusEls = [];
  }

  buildPanel(defaults: ExperimentDefaults): string {
    const showAblation = this.scenarioId === 'drone_search';
    const ablationBlock = showAblation
      ? `
        <div class="slider-row" style="margin-top:0.6rem;">
          <label for="exp-ablation">EFE ablation</label>
          <input type="checkbox" id="exp-ablation">
          <span class="slider-val">w_sal=w_nov=0, β→0.125</span>
        </div>
        <div class="info-text" style="margin-top:0.2rem;">
          Strips curiosity (w_salience=w_novelty=0) and sharpens the policy to greedy (β=0.125). Without sharpening, soft-β policy over flat EFE values is functionally random and the waypoint dispatcher rescues it. Run once off, clear results, then on, to see combined / AI / AL drop ~20–30pp while greedy and random stay flat.
        </div>`
      : '';
    return `
      <div class="panel-section">
        <h3>Experiment Controls</h3>
        <div class="slider-row">
          <label>Episodes</label>
          <input type="number" id="exp-episodes" min="5" max="500" step="5" value="${defaults.episodesPerAgent}">
          <span class="slider-val">/agent</span>
        </div>
        <div class="slider-row">
          <label>Batch size</label>
          <input type="number" id="exp-batch" min="1" max="50" step="1" value="${defaults.batchSize}">
          <span class="slider-val">live</span>
        </div>
        <div class="slider-row">
          <label>Step cap</label>
          <input type="number" id="exp-steps" min="8" max="1000" step="1" value="${defaults.stepCap}">
          <span class="slider-val">max</span>
        </div>${ablationBlock}
        <div class="info-text" style="margin-top:0.4rem;">
          Runs every agent from a fresh reset, aggregates live accuracy and reward, and updates the charts after each batch.
        </div>
        <div class="exp-status exp-status-panel" id="exp-status-panel">Ready to benchmark all agents.</div>
      </div>
      <div class="panel-section">
        <h3>Leaderboard</h3>
        <div id="exp-summary-table"></div>
      </div>
    `;
  }

  attachPanel(): void {
    const panelStatus = document.getElementById('exp-status-panel') as HTMLDivElement | null;
    this.tableEl = document.getElementById('exp-summary-table') as HTMLDivElement | null;
    this.statusEls = this.statusEls.filter(Boolean);
    if (panelStatus) this.statusEls.push(panelStatus);
    this.render();
  }

  getSettings(): { episodesPerAgent: number; batchSize: number; stepCap: number; forceExtrinsicOnly: boolean } {
    const episodesPerAgent = parseInt((document.getElementById('exp-episodes') as HTMLInputElement | null)?.value ?? '60', 10);
    const batchSize = parseInt((document.getElementById('exp-batch') as HTMLInputElement | null)?.value ?? '5', 10);
    const stepCap = parseInt((document.getElementById('exp-steps') as HTMLInputElement | null)?.value ?? '80', 10);
    const ablationEl = document.getElementById('exp-ablation') as HTMLInputElement | null;
    return {
      episodesPerAgent: Math.max(1, episodesPerAgent),
      batchSize: Math.max(1, batchSize),
      stepCap: Math.max(1, stepCap),
      forceExtrinsicOnly: !!ablationEl?.checked,
    };
  }

  getCompletedEpisodes(): number {
    const values = Array.from(this.totals.values()).map(state => state.episodes);
    return values.length === 0 ? 0 : Math.min(...values);
  }

  resetResults(): void {
    this.extraMetrics = [];
    this.trialCurves = [];
    this.trialCurveLength = 0;
    this.totals.clear();
    for (const agent of this.agents) {
      this.totals.set(agent.value, {
        episodes: 0,
        accuracySum: 0,
        rewardSum: 0,
        stepsSum: 0,
        successSum: 0,
        failureSum: 0,
        extraSums: {},
        accuracyHistory: [],
        rewardHistory: [],
        extraHistories: {},
        trialCurveSums: {},
      });
    }
    this.render();
  }

  setStatus(text: string): void {
    for (const el of this.statusEls) el.textContent = text;
  }

  applyBatch(batch: BatchResult): void {
    this.accuracyLabel = batch.accuracy_label;
    this.rewardLabel = batch.reward_label;
    this.extraMetrics = batch.extra_metrics ?? [];
    this.trialCurves = batch.trial_curves ?? [];
    if (this.trialCurves.length > 0) {
      this.trialCurveLength = batch.step_cap;
    }
    for (const agentBatch of batch.agents) {
      const state = this.totals.get(agentBatch.agent);
      if (!state) continue;
      state.episodes += agentBatch.episodes;
      state.accuracySum += agentBatch.accuracy_sum;
      state.rewardSum += agentBatch.reward_sum;
      state.stepsSum += agentBatch.steps_sum;
      state.successSum += agentBatch.success_sum;
      state.failureSum += agentBatch.failure_sum;
      for (const metric of this.extraMetrics) {
        state.extraSums[metric.key] = (state.extraSums[metric.key] ?? 0) + (agentBatch.extra_sums?.[metric.key] ?? 0);
        const history = state.extraHistories[metric.key] ?? [];
        history.push(this.computeExtraMetricValue(metric, state));
        state.extraHistories[metric.key] = history;
      }
      for (const spec of this.trialCurves) {
        const incoming = agentBatch.trial_curve_sums?.[spec.key] ?? [];
        const sums = state.trialCurveSums[spec.key] ?? new Array(this.trialCurveLength).fill(0);
        if (sums.length < this.trialCurveLength) {
          while (sums.length < this.trialCurveLength) sums.push(0);
        }
        for (let i = 0; i < incoming.length && i < sums.length; i++) {
          sums[i] += incoming[i];
        }
        state.trialCurveSums[spec.key] = sums;
      }
      state.accuracyHistory.push(state.accuracySum / state.episodes);
      state.rewardHistory.push(state.rewardSum / state.episodes);
    }
    this.render();
  }

  resize(): void {
    this.render();
  }

  private render(): void {
    this.renderCharts();
    this.renderTable();
  }

  private renderCharts(): void {
    if (!this.chartGridEl) return;

    const chartDefs = [
      {
        id: 'accuracy',
        title: this.accuracyLabel,
        historySelector: (state: AgentTotals) => state.accuracyHistory,
        formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
      },
      {
        id: 'reward',
        title: this.rewardLabel,
        historySelector: (state: AgentTotals) => state.rewardHistory,
        formatter: (value: number) => value.toFixed(2),
      },
      ...this.extraMetrics.map(metric => ({
        id: `extra-${metric.key}`,
        title: metric.label,
        historySelector: (state: AgentTotals) => state.extraHistories[metric.key] ?? [],
        formatter: (value: number) => this.formatExtraMetric(metric, value),
      })),
    ];

    const trialCurveCards = this.trialCurves.map(spec => ({
      id: `trial-${spec.key}`,
      title: spec.label,
    }));

    this.chartGridEl.innerHTML = [
      ...chartDefs.map(def => `
        <div class="exp-card">
          <h3>${def.title}</h3>
          <canvas id="exp-chart-${def.id}"></canvas>
        </div>
      `),
      ...trialCurveCards.map(def => `
        <div class="exp-card">
          <h3>${def.title}</h3>
          <canvas id="exp-chart-${def.id}"></canvas>
        </div>
      `),
    ].join('');

    for (const def of chartDefs) {
      const canvas = document.getElementById(`exp-chart-${def.id}`) as HTMLCanvasElement | null;
      this.drawLineChart(canvas, def.title, def.historySelector, def.formatter);
    }
    for (const spec of this.trialCurves) {
      const canvas = document.getElementById(`exp-chart-trial-${spec.key}`) as HTMLCanvasElement | null;
      this.drawTrialCurveChart(canvas, spec);
    }
  }

  private renderTable(): void {
    if (!this.tableEl) return;
    const showExtras = this.extraMetrics.length > 0;
    const gridTemplate = showExtras
      ? '1.2fr 1fr 1fr 1fr 1.35fr 0.8fr'
      : '1.2fr 1fr 1fr 1fr 0.8fr';
    const rows = this.agents.map(agent => {
      const state = this.totals.get(agent.value)!;
      const episodes = Math.max(state.episodes, 1);
      const accuracy = state.episodes > 0 ? state.accuracySum / episodes : 0;
      const reward = state.episodes > 0 ? state.rewardSum / episodes : 0;
      const steps = state.episodes > 0 ? state.stepsSum / episodes : 0;
      const extraSummary = this.extraMetrics.map(metric => {
        const value = this.computeExtraMetricValue(metric, state);
        return `${metric.short_label ?? metric.label}: <span>${this.formatExtraMetric(metric, value)}</span>`;
      }).join(' · ');
      return {
        agent,
        color: AGENT_COLORS[agent.value] ?? '#8080a0',
        episodes: state.episodes,
        accuracy,
        reward,
        steps,
        extraSummary,
      };
    }).sort((a, b) => b.accuracy - a.accuracy || b.reward - a.reward);

    this.tableEl.innerHTML = rows.map(row => `
      <div class="exp-row" style="--exp-grid-cols:${gridTemplate}">
        <div class="exp-agent" style="color:${row.color}">${row.agent.label}</div>
        <div class="exp-metric">${this.accuracyLabel}: <span>${(row.accuracy * 100).toFixed(1)}%</span></div>
        <div class="exp-metric">${this.rewardLabel}: <span>${row.reward.toFixed(2)}</span></div>
        <div class="exp-metric">Avg Steps: <span>${row.steps.toFixed(1)}</span></div>
        ${showExtras ? `<div class="exp-metric">${row.extraSummary}</div>` : ''}
        <div class="exp-metric">Episodes: <span>${row.episodes}</span></div>
      </div>
    `).join('');
  }

  private formatExtraMetric(metric: ExtraMetricSpec, value: number): string {
    const decimals = metric.decimals ?? 1;
    if (metric.format === 'percent') {
      return `${(value * 100).toFixed(decimals)}%`;
    }
    return value.toFixed(decimals);
  }

  private computeExtraMetricValue(metric: ExtraMetricSpec, state: AgentTotals): number {
    const numerator = state.extraSums[metric.key] ?? 0;
    if ((metric.denominator ?? 'episodes') === 'success') {
      return state.successSum > 0 ? numerator / state.successSum : 0;
    }
    return state.episodes > 0 ? numerator / state.episodes : 0;
  }

  private drawTrialCurveChart(canvas: HTMLCanvasElement | null, spec: TrialCurveSpec): void {
    if (!canvas) return;
    const card = canvas.parentElement as HTMLDivElement | null;
    const titleEl = card?.querySelector('h3');
    if (titleEl) titleEl.textContent = spec.label;
    const width = Math.max(card?.clientWidth ?? 320, 320);
    const height = 220;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    const paddingLeft = 46;
    const paddingRight = 14;
    const paddingTop = 20;
    const paddingBottom = 28;
    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    ctx.strokeStyle = '#1f2340';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = paddingTop + (chartH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(width - paddingRight, y);
      ctx.stroke();
    }

    const n = this.trialCurveLength;
    const minVal = 0;
    const maxVal = 1;

    ctx.fillStyle = '#606080';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', paddingLeft - 6, paddingTop + 4);
    ctx.fillText('0.0', paddingLeft - 6, paddingTop + chartH + 4);
    ctx.textAlign = 'center';
    ctx.fillText('Trial', paddingLeft + chartW / 2, height - 8);

    for (const agent of this.agents) {
      const state = this.totals.get(agent.value)!;
      const sums = state.trialCurveSums[spec.key];
      if (!sums || state.episodes <= 0 || n <= 0) continue;
      ctx.strokeStyle = AGENT_COLORS[agent.value] ?? '#8080a0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let t = 0; t < n; t++) {
        const value = (sums[t] ?? 0) / state.episodes;
        const x = paddingLeft + (chartW * t) / Math.max(n - 1, 1);
        const yNorm = (value - minVal) / Math.max(maxVal - minVal, 1e-6);
        const y = paddingTop + chartH - yNorm * chartH;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Compact legend
    let legendX = paddingLeft;
    const legendY = 10;
    ctx.textAlign = 'left';
    for (const agent of this.agents) {
      ctx.fillStyle = AGENT_COLORS[agent.value] ?? '#8080a0';
      ctx.fillRect(legendX, legendY - 7, 10, 10);
      ctx.fillStyle = '#a0a0c0';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(agent.label, legendX + 14, legendY + 1);
      legendX += 86;
    }
  }

  private drawLineChart(
    canvas: HTMLCanvasElement | null,
    title: string,
    historySelector: (state: AgentTotals) => number[],
    valueFormatter: (value: number) => string = (value: number) => value.toFixed(2),
  ): void {
    if (!canvas) return;
    const card = canvas.parentElement as HTMLDivElement | null;
    const titleEl = card?.querySelector('h3');
    if (titleEl) titleEl.textContent = title;
    const width = Math.max(card?.clientWidth ?? 320, 320);
    const height = 220;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    const paddingLeft = 46;
    const paddingRight = 14;
    const paddingTop = 20;
    const paddingBottom = 28;
    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    ctx.strokeStyle = '#1f2340';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = paddingTop + (chartH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(width - paddingRight, y);
      ctx.stroke();
    }

    const allValues = Array.from(this.totals.values()).flatMap(historySelector);
    const maxLen = Math.max(1, ...Array.from(this.totals.values()).map(state => historySelector(state).length));
    let minVal = 0;
    let maxVal = 1;
    if (allValues.length > 0) {
      minVal = Math.min(...allValues);
      maxVal = Math.max(...allValues);
      if (Math.abs(maxVal - minVal) < 1e-6) {
        maxVal = minVal + 1.0;
      }
    }

    ctx.fillStyle = '#606080';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
  ctx.fillText(valueFormatter(maxVal), paddingLeft - 6, paddingTop + 4);
  ctx.fillText(valueFormatter(minVal), paddingLeft - 6, paddingTop + chartH + 4);
    ctx.textAlign = 'center';
    ctx.fillText('Batches', paddingLeft + chartW / 2, height - 8);

    for (const agent of this.agents) {
      const state = this.totals.get(agent.value)!;
      const history = historySelector(state);
      if (history.length === 0) continue;
      ctx.strokeStyle = AGENT_COLORS[agent.value] ?? '#8080a0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((value, index) => {
        const x = paddingLeft + (chartW * index) / Math.max(maxLen - 1, 1);
        const yNorm = (value - minVal) / Math.max(maxVal - minVal, 1e-6);
        const y = paddingTop + chartH - yNorm * chartH;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Compact legend
    let legendX = paddingLeft;
    const legendY = 10;
    ctx.textAlign = 'left';
    for (const agent of this.agents) {
      ctx.fillStyle = AGENT_COLORS[agent.value] ?? '#8080a0';
      ctx.fillRect(legendX, legendY - 7, 10, 10);
      ctx.fillStyle = '#a0a0c0';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(agent.label, legendX + 14, legendY + 1);
      legendX += 86;
    }
  }
}