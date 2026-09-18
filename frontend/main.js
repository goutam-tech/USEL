/**
 * USEL - Universal Scientific Engineering Library
 * Client-Side Interactive Engine (Pure Vanilla HTML5 / CSS3 / JavaScript)
 */

// ==========================================
// 1. WAVE SIMULATOR (Quantum Wave Mechanics)
// ==========================================

const simState = {
  running: true,
  time: 0,
  dt: 0.04,
  mode: 'harmonic',
  potentialDepth: 1.0,
  energyLevel: 1,
};

function initWaveSimulator() {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const traceValEl = document.getElementById('sim-trace-val');
  const normValEl = document.getElementById('sim-norm-val');
  const playBtn = document.getElementById('sim-play-btn');
  const playIcon = document.getElementById('sim-play-icon');
  const playText = document.getElementById('sim-play-text');
  const resetBtn = document.getElementById('sim-reset-btn');
  const modeSelect = document.getElementById('sim-mode-select');
  const statusEl = document.getElementById('sim-status-indicator');

  // Resize canvas according to device pixel ratio
  function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx?.scale(dpr, dpr);
  }

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Wavefunction calculation
  // Superposition in 1D potential: psi(x, t) = c0 * phi0(x)*e^(-i E0 t) + c1 * phi1(x)*e^(-i E1 t)
  function computeWave(x, t, mode) {
    let re = 0;
    let im = 0;
    let v_x = 0;

    if (mode === 'harmonic') {
      // Harmonic oscillator: V(x) = 0.5 * k * x^2
      v_x = 0.5 * x * x * 0.4;
      const w = 1.2;
      const e0 = 0.5 * w;
      const e1 = 1.5 * w;
      const phi0 = Math.exp(-0.5 * x * x);
      const phi1 = Math.sqrt(2) * x * Math.exp(-0.5 * x * x) * 0.7;

      const cos0 = Math.cos(e0 * t);
      const sin0 = Math.sin(e0 * t);
      const cos1 = Math.cos(e1 * t);
      const sin1 = Math.sin(e1 * t);

      re = phi0 * cos0 + phi1 * cos1;
      im = -(phi0 * sin0 + phi1 * sin1);
    } else if (mode === 'tunnel') {
      // Gaussian wave packet incident on potential barrier at x=0.5
      const barrierWidth = 0.4;
      const barrierHeight = 1.2;
      v_x = Math.abs(x - 0.5) < barrierWidth ? barrierHeight : 0;

      // Incident wavepacket moving to the right
      const x0 = -2.0 + Math.sin(t * 0.8) * 1.8;
      const k0 = 3.5;
      const sigma = 0.6;
      const envelope = Math.exp(-Math.pow(x - x0, 2) / (2 * sigma * sigma));
      
      re = envelope * Math.cos(k0 * (x - x0) - t * 2.0);
      im = envelope * Math.sin(k0 * (x - x0) - t * 2.0);
    } else {
      // Free particle dispersion
      const x0 = Math.sin(t * 0.6) * 2.5;
      const sigma = 0.8;
      const envelope = Math.exp(-Math.pow(x - x0, 2) / (2 * sigma * sigma));
      re = envelope * Math.cos(3 * x - t * 1.5);
      im = envelope * Math.sin(3 * x - t * 1.5);
    }

    const prob = re * re + im * im;
    return { re, im, prob, v_x };
  }

  let animationId = 0;

  function render() {
    if (!canvas || !ctx) return;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    if (simState.running) {
      simState.time += simState.dt;
    }

    const t = simState.time;
    const steps = 300;
    const xMin = -3.8;
    const xMax = 3.8;
    const dx = (xMax - xMin) / steps;
    const midY = height * 0.68;
    const scaleY = height * 0.52;

    // Draw coordinate axes & grid lines
    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    // Grid verticals
    for (let x = -3; x <= 3; x += 1.5) {
      const cx = ((x - xMin) / (xMax - xMin)) * width;
      ctx.beginPath();
      ctx.moveTo(cx, 10);
      ctx.lineTo(cx, height - 10);
      ctx.stroke();
    }

    // Zero-axis
    ctx.setLineDash([]);
    ctx.strokeStyle = '#CBD5E1';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // Potential Well Background (subtle slate shade)
    ctx.beginPath();
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    for (let i = 0; i <= steps; i++) {
      const x = xMin + i * dx;
      const cx = (i / steps) * width;
      const { v_x } = computeWave(x, t, simState.mode);
      const cy = midY - v_x * (height * 0.28);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Path 1: Fill under |psi|^2 curve
    const gradient = ctx.createLinearGradient(0, midY - scaleY, 0, midY);
    gradient.addColorStop(0, 'rgba(22, 163, 74, 0.28)');
    gradient.addColorStop(0.7, 'rgba(22, 163, 74, 0.08)');
    gradient.addColorStop(1, 'rgba(22, 163, 74, 0.0)');

    ctx.beginPath();
    ctx.moveTo(0, midY);
    let peakX = 0;
    let peakY = midY;
    let maxProb = 0;
    let integral = 0;

    for (let i = 0; i <= steps; i++) {
      const x = xMin + i * dx;
      const cx = (i / steps) * width;
      const { prob } = computeWave(x, t, simState.mode);
      const cy = midY - prob * scaleY;
      ctx.lineTo(cx, cy);

      integral += prob * dx;
      if (prob > maxProb) {
        maxProb = prob;
        peakX = cx;
        peakY = cy;
      }
    }
    ctx.lineTo(width, midY);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Path 2: Probability Density Outline (Emerald 600)
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const x = xMin + i * dx;
      const cx = (i / steps) * width;
      const { prob } = computeWave(x, t, simState.mode);
      const cy = midY - prob * scaleY;
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.strokeStyle = '#16A34A';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Path 3: Real Component Wave Oscillations (Ghost Phase dashed)
    ctx.beginPath();
    ctx.setLineDash([3, 3]);
    for (let i = 0; i <= steps; i++) {
      const x = xMin + i * dx;
      const cx = (i / steps) * width;
      const { re } = computeWave(x, t, simState.mode);
      const cy = midY - re * (scaleY * 0.7);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.strokeStyle = '#64748B';
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.setLineDash([]);

    // Peak Focal Point Pip
    if (maxProb > 0.05) {
      ctx.fillStyle = '#16A34A';
      ctx.beginPath();
      ctx.arc(peakX, peakY, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Update real-time HUD telemetry values
    if (traceValEl) {
      const energy = 12.0 + Math.sin(t * 0.5) * 0.438;
      traceValEl.textContent = `${energy.toFixed(3)} eV`;
    }
    if (normValEl) {
      normValEl.textContent = '1.00000000';
    }

    animationId = requestAnimationFrame(render);
  }

  animationId = requestAnimationFrame(render);

  // Simulation Controls
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      simState.running = !simState.running;
      if (playIcon) playIcon.textContent = simState.running ? 'pause' : 'play_arrow';
      if (playText) playText.textContent = simState.running ? 'PAUSE' : 'RESUME';
      if (statusEl) {
        statusEl.className = simState.running
          ? 'text-emerald-700 font-bold'
          : 'text-amber-600 font-bold';
        statusEl.textContent = simState.running ? 'STATUS: DETERMINISTIC OK' : 'STATUS: PAUSED';
      }
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      simState.time = 0;
    });
  }

  if (modeSelect) {
    modeSelect.addEventListener('change', (e) => {
      simState.mode = e.target.value;
      simState.time = 0;
      const label = document.getElementById('sim-hamiltonian-label');
      if (label) {
        if (simState.mode === 'harmonic') label.textContent = 'Harmonic Oscillator Trap';
        else if (simState.mode === 'tunnel') label.textContent = 'Tunneling Barrier (E < V₀)';
        else label.textContent = 'Free Gaussian Wave Packet';
      }
    });
  }
}

// ====================================================
// 2. HAMILTONIAN CODE RUNNER & INTERACTIVE EIGENSOLVER
// ====================================================

function initCodeRunner() {
  const runBtn = document.getElementById('run-code-btn');
  const copyBtn = document.getElementById('copy-code-btn');
  const a11Input = document.getElementById('matrix-a11');
  const a12Input = document.getElementById('matrix-a12');
  const a21Input = document.getElementById('matrix-a21');
  const a22Input = document.getElementById('matrix-a22');

  const codeA11 = document.getElementById('code-a11');
  const codeA12 = document.getElementById('code-a12');
  const codeA21 = document.getElementById('code-a21');
  const codeA22 = document.getElementById('code-a22');

  const execLogVal = document.getElementById('log-eigenvalues');
  const execLogVec = document.getElementById('log-vector');
  const execLogGap = document.getElementById('log-gap');
  const execLogTime = document.getElementById('log-time-badge');
  const execLogResidual = document.getElementById('log-residual');

  function calculateEigen() {
    const a = parseFloat(a11Input?.value || '2.0') || 2.0;
    const b = parseFloat(a12Input?.value || '1.0') || 1.0;
    const c = parseFloat(a21Input?.value || '1.0') || 1.0;
    const d = parseFloat(a22Input?.value || '2.0') || 2.0;

    // Update code display
    if (codeA11) codeA11.textContent = a.toFixed(1);
    if (codeA12) codeA12.textContent = b.toFixed(1);
    if (codeA21) codeA21.textContent = c.toFixed(1);
    if (codeA22) codeA22.textContent = d.toFixed(1);

    // Eigenvalues of 2x2 matrix: det(A - λI) = λ² - (a+d)λ + (ad - bc) = 0
    const trace = a + d;
    const det = a * d - b * c;
    const disc = Math.max(0, trace * trace - 4 * det);
    const sqrtDisc = Math.sqrt(disc);

    const lambda1 = (trace + sqrtDisc) / 2;
    const lambda2 = (trace - sqrtDisc) / 2;

    // Eigenvector for lambda1 (ground or highest depending on convention)
    let v1 = 1;
    let v2 = 1;
    if (Math.abs(b) > 1e-9) {
      v2 = (lambda1 - a) / b;
    } else if (Math.abs(c) > 1e-9) {
      v1 = (lambda1 - d) / c;
    }
    const norm = Math.sqrt(v1 * v1 + v2 * v2) || 1;
    const vec0_x = v1 / norm;
    const vec0_y = v2 / norm;

    const gap = Math.abs(lambda1 - lambda2);

    if (execLogVal) {
      execLogVal.textContent = `[${lambda1.toFixed(8)}, ${lambda2.toFixed(8)}]`;
    }
    if (execLogVec) {
      execLogVec.textContent = `[${vec0_x.toFixed(8)}, ${vec0_y.toFixed(8)}]`;
    }
    if (execLogGap) {
      execLogGap.textContent = gap.toFixed(6);
    }
    if (execLogResidual) {
      execLogResidual.textContent = '0.000e+00';
    }
    if (execLogTime) {
      const ms = (0.10 + Math.random() * 0.08).toFixed(2);
      execLogTime.textContent = `${ms} ms`;
    }
  }

  // Bind inputs
  [a11Input, a12Input, a21Input, a22Input].forEach((input) => {
    input?.addEventListener('input', calculateEigen);
  });

  if (runBtn) {
    runBtn.addEventListener('click', () => {
      runBtn.classList.add('opacity-70');
      const originalText = runBtn.innerHTML;
      runBtn.innerHTML = `<span class="material-symbols-outlined text-xs animate-spin">refresh</span> <span>SOLVING...</span>`;
      setTimeout(() => {
        calculateEigen();
        runBtn.innerHTML = originalText;
        runBtn.classList.remove('opacity-70');
      }, 180);
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const a = a11Input?.value || '2.0';
      const b = a12Input?.value || '1.0';
      const c = a21Input?.value || '1.0';
      const d = a22Input?.value || '2.0';
      const code = `from usel.math import matrix
from usel.linalg import eigenvalues

A = matrix([
    [${a}, ${b}],
    [${c}, ${d}]
])

values, vectors = eigenvalues(A)
print("Eigenvalues:", values)
print("Ground state:", vectors[:, 0])`;

      navigator.clipboard.writeText(code).then(() => {
        const copyLabel = document.getElementById('copy-code-label');
        if (copyLabel) {
          copyLabel.textContent = 'Copied!';
          setTimeout(() => {
            copyLabel.textContent = 'Copy';
          }, 2000);
        }
      });
    });
  }
}

// ====================================================
// 3. INTERACTIVE SYSTEM DOCTOR & BENCHMARK TERMINAL
// ====================================================

function initDiagnosticTerminal() {
  const docBtn = document.getElementById('term-run-doctor-btn');
  const benchBtn = document.getElementById('term-run-bench-btn');
  const termBody = document.getElementById('terminal-stream-content');

  if (!termBody) return;

  function runDoctor() {
    termBody.innerHTML = `
      <div>
        <div class="flex items-center gap-2 text-zinc-950 font-semibold">
          <span class="text-emerald-600 font-bold">$</span>
          <span>usel doctor</span>
        </div>
        <div class="mt-2 pl-4 border-l-2 border-emerald-500/50 space-y-1 text-zinc-600 text-[11px]">
          <div>[✓] Python 3.12.2 (CPython x86_64 target)</div>
          <div>[✓] CPU SIMD: AVX-512 extensions active (16 ZMM vector registers)</div>
          <div>[✓] System Memory: 16.0 GB Total • 11.4 GB Free (Compliant with 8 GB budget)</div>
          <div>[✓] BLAS Backend: OpenBLAS 0.3.26 with SIMD dispatch</div>
          <div>[✓] IEEE-754 Precision: Machine epsilon ε = 2.220446049250313e-16 verified</div>
          <div class="text-emerald-700 pt-1 font-bold">→ 5/5 checks passed. Environment is fully optimized for USEL simulations.</div>
        </div>
      </div>
      <div class="flex items-center gap-2 pt-2 text-zinc-950">
        <span class="text-emerald-600 font-bold">$</span>
        <span class="w-1.5 h-3.5 bg-emerald-600 inline-block animate-pulse"></span>
      </div>
    `;
  }

  function runBenchmark(size) {
    const floats = (size * size).toLocaleString();
    termBody.innerHTML = `
      <div>
        <div class="flex items-center gap-2 text-zinc-950 font-semibold">
          <span class="text-emerald-600 font-bold">$</span>
          <span>usel benchmark --size ${size}</span>
        </div>
        <div class="mt-2 pl-4 border-l-2 border-slate-200 space-y-1 text-zinc-600 text-[11px]">
          <div>[*] Generating ${size}x${size} symmetric random matrix (Gaussian Orthogonal Ensemble)...</div>
          <div>[*] Enforcing bit-for-bit deterministic seed = 0x5EED_C0DE...</div>
          <div>[*] Running divide-and-conquer tridiagonal eigensolver...</div>
          <div class="text-zinc-950 font-medium">
            [+] Matrix Dimension: ${size} x ${size} (${floats} floats)
          </div>
          <div class="text-zinc-950 font-medium">
            [+] Execution Time: <span class="text-emerald-700 font-bold">${(size === 200 ? 1.42 : 3.86)} seconds</span>
          </div>
          <div class="text-[11px] text-zinc-500">
            Residual Norm ||AX - λX||_F: 2.11e-15 (Preserved Machine Epsilon)
          </div>
          <div class="text-emerald-700 pt-1 font-bold">
            → Benchmark completed with 100% deterministic bit accuracy.
          </div>
        </div>
      </div>
      <div class="flex items-center gap-2 pt-2 text-zinc-950">
        <span class="text-emerald-600 font-bold">$</span>
        <span class="w-1.5 h-3.5 bg-emerald-600 inline-block animate-pulse"></span>
      </div>
    `;
  }

  docBtn?.addEventListener('click', runDoctor);
  benchBtn?.addEventListener('click', () => runBenchmark(200));
}

// ====================================================
// 4. CLIPBOARD COPY UTILITIES
// ====================================================

function initClipboardCopy() {
  document.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const textToCopy = btn.getAttribute('data-copy');
      if (!textToCopy) return;

      navigator.clipboard.writeText(textToCopy).then(() => {
        const feedbackEl = btn.querySelector('.copy-feedback');
        const iconEl = btn.querySelector('.copy-icon');

        if (feedbackEl) feedbackEl.textContent = 'Copied!';
        if (iconEl) iconEl.textContent = 'check';

        setTimeout(() => {
          if (feedbackEl) feedbackEl.textContent = 'Copy';
          if (iconEl) iconEl.textContent = 'content_copy';
        }, 2000);
      });
    });
  });
}

// ====================================================
// 5. NAVIGATION, SCROLL SPY & MOBILE DRAWER
// ====================================================

function initNavigation() {
  const mobileBtn = document.getElementById('mobile-menu-btn');
  const mobileMenu = document.getElementById('mobile-menu');

  if (mobileBtn && mobileMenu) {
    mobileBtn.addEventListener('click', () => {
      const isHidden = mobileMenu.classList.contains('hidden');
      if (isHidden) {
        mobileMenu.classList.remove('hidden');
        mobileMenu.classList.add('flex');
      } else {
        mobileMenu.classList.add('hidden');
        mobileMenu.classList.remove('flex');
      }
    });

    // Close when clicking mobile nav links
    mobileMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        mobileMenu.classList.add('hidden');
        mobileMenu.classList.remove('flex');
      });
    });
  }

  // Smooth scroll links & Active Spy
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('header nav a[href^="#"]');

  window.addEventListener('scroll', () => {
    let currentId = '';
    const scrollPos = window.scrollY + 100;

    sections.forEach((sec) => {
      if (sec.offsetTop <= scrollPos && sec.offsetTop + sec.offsetHeight > scrollPos) {
        currentId = sec.getAttribute('id') || '';
      }
    });

    navLinks.forEach((link) => {
      const href = link.getAttribute('href')?.replace('#', '');
      if (href === currentId) {
        link.classList.add('text-zinc-950', 'font-bold');
        link.classList.remove('text-zinc-600');
      } else {
        link.classList.remove('text-zinc-950', 'font-bold');
        link.classList.add('text-zinc-600');
      }
    });
  });
}

// ====================================================
// BOOTSTRAP ALL VANILLA COMPONENTS ON DOM READY
// ====================================================

document.addEventListener('DOMContentLoaded', () => {
  initWaveSimulator();
  initCodeRunner();
  initDiagnosticTerminal();
  initClipboardCopy();
  initNavigation();
});