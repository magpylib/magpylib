/*
 * Magnetic wordmark for the navbar brand.
 *
 * Every letter of "Magpylib" beside the logo is treated as a magnetic dipole and
 * the pointer as a bar magnet, so the letters obey the same physics the library
 * itself computes: the force is the gradient of the interaction energy,
 * F = grad(m . B), and the torque is m x B, which is what makes them swivel to
 * line up with the field rather than merely slide toward the pointer. Poles are
 * tinted with magpylib's own magnetization colours (see defaults_values.py).
 *
 * Move onto a letter and it latches to the pointer and comes along; shake to throw
 * it off. Hold on to it too long and the magpie lifts off the logo, chases the
 * letter down, takes it back and returns it to the wordmark before flying home.
 * The letters always spring back to the baseline, and their travel is capped so
 * the brand stays readable and the link underneath stays easy to hit.
 *
 * Styling lives in _static/custom.css (.mgw / .mgl).
 */
(() => {
  "use strict";

  const WORD = "Magpylib";

  /* Motion is the entire point of this, so there is nothing to degrade to. */
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const P = {
    /* absolute (px/s^2, 1/s) */
    strength: 300, spring: 320, damp: 12, torque: 80,
    rspring: 160, rdamp: 9,
    maxForce: 9000, maxSpring: 18000, maxSpeed: 2600,
    /* how a latched letter is carried */
    grip: 900, gripDamp: 46, gripTorque: 420, gripRDamp: 30,
    fling: 0.5, cooldown: 0.55, maxHeld: 1,
    /* the magpie: how long it tolerates a theft, and how it flies */
    patience: 1.8, birdAccel: 3000, birdDamp: 3.2, birdMax: 1000,
    birdCatch: 26, birdDrop: 12, birdCool: 0.9, birdScale: 2.2,
    flapRate: 32, flapBase: -2, flapSweep: 36,
    /* a shake is counted in direction reversals, not raw speed, so dragging a
       letter quickly across the page keeps it and only a wiggle sheds it */
    shakeSpeed: 380, shakeDecay: 2.2, shakeOff: 2.6,
    /* derived from the rendered font size in measure() */
    soft: 26, reach: 96, limit: 7, gap: 30, lead: 22, dHalf: 40,
  };

  let host = null, cRect = null;
  const letters = [], held = [];
  let MX = -1e5, MY = -1e5, active = false;
  let cvx = 0, cvy = 0, lastT = 0, lastX = 0, lastY = 0, shake = 0;

  /* Where the magpie sits inside the logo image, as fractions of it. Kept in step
     with _static/images/magpylib_logo_bird.png, which is that same region cropped
     out, and with ..._nobird.png, which is the logo without it. */
  const BIRD_BOX = { cx: 0.1906, cy: 0.2611, w: 0.3811, h: 0.5222 };
  const bird = { el:null, wing:null, mode:"perched", target:null,
                 x:0, y:0, vx:0, vy:0, w:0, h:0, t:0, face:1, scale:1 };
  let holdTime = 0, logoImgs = [], srcPerched = [], srcFlown = [];

  /* ---------- split the brand text into letters ---------- */
  function build(){
    host = document.querySelector(".navbar-brand .title, .navbar-brand .logo__title");
    if (!host || host.textContent.trim() !== WORD) return false;

    const wrap = document.createElement("span");
    wrap.className = "mgw";
    wrap.setAttribute("aria-label", WORD);
    for (const ch of WORD){
      const el = document.createElement("span");
      el.className = "mgl";
      el.setAttribute("aria-hidden", "true");
      el.dataset.ch = ch;
      el.textContent = ch;
      wrap.appendChild(el);
      letters.push({ el, hx:0, hy:0, ox:0, oy:0, hw:0, hh:0,
                     x:0, y:0, vx:0, vy:0, a:0, w:0, p:0,
                     stuck:false, carried:false, cool:0, attach:0 });
    }
    host.textContent = "";
    host.appendChild(wrap);

    bird.el = document.createElement("div");
    bird.el.className = "mag-bird";
    bird.el.setAttribute("aria-hidden", "true");
    /* The logo magpie is one flat silhouette with no wing to animate, so the wing
       is a separate shape in the same ink, hinged at the shoulder. Both sit under
       one drop-shadow, so they read as a single beating silhouette. */
    bird.wing = document.createElement("div");
    bird.wing.className = "mag-bird__wing";
    bird.el.appendChild(bird.wing);
    document.body.appendChild(bird.el);

    /* Swapping the logo for its bird-less twin is what actually makes the magpie
       leave its perch. If the file is named something else the chase still works,
       the logo simply keeps its bird. */
    logoImgs = Array.from(document.querySelectorAll(".navbar-brand img.logo__image"));
    srcPerched = logoImgs.map(img => img.getAttribute("src"));
    /* Sphinx copies html_logo to the root of _static, while html_static_path keeps
       our twin under _static/images, so the sibling path depends on which one the
       theme ended up pointing at. */
    srcFlown = srcPerched.map(src => /\/images\/[^/]*$/.test(src)
      ? src.replace(/magpylib_logo\.png$/, "magpylib_logo_nobird.png")
      : src.replace(/magpylib_logo\.png$/, "images/magpylib_logo_nobird.png"));
    if (srcFlown.some((src, i) => src === srcPerched[i])) srcFlown = [];
    /* if that guess is wrong the chase still happens, the logo just keeps its bird */
    srcFlown.forEach(src => { const pre = new Image(); pre.onerror = () => { srcFlown = []; }; pre.src = src; });
    return true;
  }

  /* ---------- the magpie ---------- */
  function perch(){
    /* the theme ships a light and a dark logo and hides one of them */
    for (const img of logoImgs){
      const r = img.getBoundingClientRect();
      if (r.width) return { x: r.left + BIRD_BOX.cx * r.width, y: r.top + BIRD_BOX.cy * r.height,
                            w: BIRD_BOX.w * r.width, h: BIRD_BOX.h * r.height };
    }
    return null;
  }

  function showLogoBird(on){
    if (!srcFlown.length) return;
    logoImgs.forEach((img, i) => img.setAttribute("src", on ? srcPerched[i] : srcFlown[i]));
  }

  function launch(target){
    const p = perch();
    if (!p) return;
    bird.x = p.x; bird.y = p.y; bird.w = p.w; bird.h = p.h;
    bird.vx = 0; bird.vy = 0; bird.t = 0; bird.scale = 1;
    bird.mode = "hunt"; bird.target = target;
    bird.el.style.width = p.w.toFixed(1) + "px";
    bird.el.style.height = p.h.toFixed(1) + "px";
    bird.el.style.opacity = "1";
    showLogoBird(false);
  }

  function land(){
    bird.mode = "perched"; bird.target = null;
    bird.el.style.opacity = "0";
    showLogoBird(true);
  }

  function seek(tx, ty, dt){
    const dx = tx - bird.x, dy = ty - bird.y;
    const d = Math.hypot(dx, dy) || 1e-6;
    bird.vx += (dx/d * P.birdAccel - bird.vx * P.birdDamp) * dt;
    bird.vy += (dy/d * P.birdAccel - bird.vy * P.birdDamp) * dt;
    const sp = Math.hypot(bird.vx, bird.vy);
    if (sp > P.birdMax){ bird.vx *= P.birdMax/sp; bird.vy *= P.birdMax/sp; }
    bird.x += bird.vx*dt; bird.y += bird.vy*dt;
    return d;
  }

  function updateBird(dt){
    if (bird.mode === "perched") return;
    bird.t += dt;

    if (bird.mode === "hunt"){
      const L = bird.target;
      if (!L || !L.stuck){
        bird.mode = "home"; bird.target = null;      // shaken off before it arrived
      } else if (seek(L.hx + L.x, L.hy + L.y, dt) < P.birdCatch){
        const i = held.indexOf(L);
        if (i >= 0) held.splice(i, 1);
        L.stuck = false; L.carried = true; L.cool = Infinity;
        holdTime = 0; shake = 0;
        bird.mode = "carry";
      }
    }
    if (bird.mode === "carry"){
      const L = bird.target;
      if (seek(L.hx, L.hy - bird.h*0.40, dt) < P.birdDrop){
        L.carried = false; L.cool = P.birdCool;      // a moment before it can be taken again
        bird.target = null; bird.mode = "home";
      }
    }
    if (bird.mode === "home"){
      const p = perch();
      if (!p){ land(); return; }
      bird.w = p.w; bird.h = p.h;
      if (seek(p.x, p.y, dt) < 7){ land(); return; }
    }

    /* it grows as it leaves the logo, so it still reads while crossing the page */
    const want = bird.mode === "home" ? 1 : P.birdScale;
    bird.scale += (want - bird.scale) * Math.min(1, dt * 5);

    /* one flap drives both the wing and the body: it rises on the downstroke */
    const flap = Math.sin(bird.t * P.flapRate);
    bird.wing.style.transform = "rotate(" + (P.flapBase + flap * P.flapSweep).toFixed(1) + "deg)";

    const bob = -flap * bird.h * 0.055;
    if (Math.abs(bird.vx) > 40) bird.face = bird.vx > 0 ? -1 : 1;   // the sprite faces left
    const tilt = Math.max(-0.45, Math.min(0.45, bird.vy / 1100)) * bird.face;
    bird.el.style.transform =
      "translate3d(" + (bird.x - bird.w/2).toFixed(1) + "px," + (bird.y - bird.h/2 + bob).toFixed(1) + "px,0)" +
      " scale(" + (bird.scale * bird.face).toFixed(3) + "," + bird.scale.toFixed(3) + ")" +
      " rotate(" + tilt.toFixed(3) + "rad)";
  }

  /* Letter homes are kept as offsets inside the brand, so the sticky header can
     move without re-measuring every glyph. */
  function measure(){
    for (const L of letters) L.el.style.transform = "none";
    cRect = host.getBoundingClientRect();
    if (!cRect.width) return;
    /* scale off the font size, not the line box, so line-height cannot change the feel */
    const fs = parseFloat(getComputedStyle(host).fontSize) || 16;
    for (const L of letters){
      const r = L.el.getBoundingClientRect();
      L.ox = r.left + r.width/2  - cRect.left;
      L.oy = r.top  + r.height/2 - cRect.top;
      L.hw = r.width * 0.50;      // the glyph band, so "on a letter" means on the glyph
      L.hh = fs * 0.42;
    }
    P.soft  = fs * 1.60;
    P.reach = fs * 6.00;
    P.limit = fs * 0.28;   // max travel: a nudge, not a leap
    P.dHalf = fs * 2.20;   // distance at which a letter reads half-magnetised
    P.gap   = fs * 1.90;
    P.lead  = fs * 1.40;
    render();
  }

  /* ---------- pointer ---------- */
  function onMove(e){
    MX = e.clientX; MY = e.clientY; active = true;

    /* A shake is a reversal of direction at speed — dragging a letter fast in a
       straight line must not shed it. */
    const t = performance.now(), dt = Math.max(4, t - lastT) / 1000;
    const nvx = (MX - lastX)/dt, nvy = (MY - lastY)/dt;
    if (lastT){
      if (Math.hypot(nvx, nvy) > P.shakeSpeed && nvx*cvx + nvy*cvy < 0) shake += 1;
    }
    cvx = nvx; cvy = nvy; lastT = t; lastX = MX; lastY = MY;
  }

  function dropAll(){
    for (const L of held){
      L.stuck = false; L.cool = P.cooldown;
      L.vx += cvx * P.fling; L.vy += cvy * P.fling;
      L.w  += (Math.random() - 0.5) * 12;
      L.a = ((L.a + Math.PI) % (Math.PI*2) + Math.PI*2) % (Math.PI*2) - Math.PI;
    }
    held.length = 0; shake = 0; holdTime = 0;
  }

  function tryCatch(){
    if (!active) return;
    for (const L of letters){
      if (L.stuck || L.cool > 0 || held.length >= P.maxHeld) continue;
      const dx = MX - (L.hx + L.x), dy = MY - (L.hy + L.y);
      const ca = Math.cos(L.a), sa = Math.sin(L.a);
      const lx =  dx*ca + dy*sa;          // pointer in the letter's own frame
      const ly = -dx*sa + dy*ca;
      if (Math.abs(lx) < L.hw && Math.abs(ly) < L.hh){ L.stuck = true; held.push(L); }
    }
  }

  /* ---------- field of the pointer magnet (moment points up) ---------- */
  let BX = 0, BY = 0;
  function fieldAt(px, py, C){
    const rx = px - MX, ry = py - MY;
    const d2 = rx*rx + ry*ry + P.soft*P.soft;
    const d = Math.sqrt(d2);
    const ux = rx/d, uy = ry/d;
    const md = -uy;                        // moment is (0,-1), so m.u = -uy
    const k = C/(d2*d);
    BX = k*(3*md*ux);
    BY = k*(3*md*uy + 1);
  }

  function step(dt){
    const C  = P.strength * Math.pow(P.reach, 4) / 3;
    const B0 = C / (P.dHalf*P.dHalf*P.dHalf);   // sets how close is "fully magnetised"
    const e = 1;

    let run = P.lead;
    for (const L of held){ L.attach = run + P.gap*0.5; run += P.gap; }

    for (const L of letters){
      if (L.cool > 0) L.cool -= dt;

      if (L.stuck || L.carried){
        let tx, ty;
        if (L.carried){
          tx = bird.x; ty = bird.y + bird.h*0.42;    // clutched under the magpie
        } else {
          /* hang off the pointer, but never off the edge of the window */
          const padX = L.hw + 6, padY = L.hh + 6;
          tx = Math.max(padX, Math.min(innerWidth  - padX, MX));
          ty = Math.max(padY, Math.min(innerHeight - padY, MY + L.attach));
        }
        L.vx += (P.grip*(tx - L.hx - L.x) - P.gripDamp*L.vx) * dt;
        L.vy += (P.grip*(ty - L.hy - L.y) - P.gripDamp*L.vy) * dt;
        L.x += L.vx*dt; L.y += L.vy*dt;
        let da = (-L.a + Math.PI) % (Math.PI*2); if (da < 0) da += Math.PI*2; da -= Math.PI;
        L.w += (P.gripTorque*da - P.gripRDamp*L.w) * dt;
        L.a += L.w*dt;
        L.p += (1 - L.p) * Math.min(1, dt*16);
        continue;
      }

      let fx = 0, fy = 0, tq = 0, pol = 0;
      if (active){
        const px = L.hx + L.x, py = L.hy + L.y;
        const ca = Math.cos(L.a), sa = Math.sin(L.a);
        const lmx = sa, lmy = -ca;                        // this letter's moment
        fieldAt(px+e, py, C); const a1 = lmx*BX + lmy*BY;
        fieldAt(px-e, py, C); const a2 = lmx*BX + lmy*BY;
        fieldAt(px, py+e, C); const a3 = lmx*BX + lmy*BY;
        fieldAt(px, py-e, C); const a4 = lmx*BX + lmy*BY;
        fx = (a1-a2)/(2*e); fy = (a3-a4)/(2*e);           // F = grad(m.B)
        const fm = Math.hypot(fx, fy);
        if (fm > P.maxForce){ fx *= P.maxForce/fm; fy *= P.maxForce/fm; }
        fieldAt(px, py, C);
        const bm = Math.hypot(BX, BY) || 1e-9;
        tq  = P.torque * ((ca*BX + sa*BY)/bm) * (bm/(bm+B0));   // aligns m with B
        pol = bm/(bm+B0);
      }

      /* The restoring spring stiffens with displacement, so a letter can be pulled
         hard but never far enough to scramble the brand or shift the link target.
         Both it and the speed are capped: a letter dropped across the page would
         otherwise make this spring explode. */
      const off = Math.hypot(L.x, L.y) / P.limit;
      const stiff = 1 + Math.min(off*off, 36);
      let sx = -P.spring*L.x*stiff, sy = -P.spring*L.y*stiff;
      const sm = Math.hypot(sx, sy);
      if (sm > P.maxSpring){ sx *= P.maxSpring/sm; sy *= P.maxSpring/sm; }
      L.vx += (fx + sx - P.damp*L.vx) * dt;
      L.vy += (fy + sy - P.damp*L.vy) * dt;
      const sp = Math.hypot(L.vx, L.vy);
      if (sp > P.maxSpeed){ L.vx *= P.maxSpeed/sp; L.vy *= P.maxSpeed/sp; }
      L.x += L.vx*dt; L.y += L.vy*dt;
      L.w += (tq - P.rspring*L.a - P.rdamp*L.w) * dt;
      if (L.w > 40) L.w = 40; else if (L.w < -40) L.w = -40;
      L.a += L.w*dt;
      L.p += (pol - L.p) * Math.min(1, dt*12);
    }
  }

  function render(){
    for (const L of letters){
      L.el.style.transform = "translate3d(" + L.x.toFixed(2) + "px," + L.y.toFixed(2) + "px,0) rotate(" + L.a.toFixed(4) + "rad)";
      L.el.style.setProperty("--p", (L.p * 0.85).toFixed(3));
    }
  }

  const atRest = () => letters.every(L =>
    Math.abs(L.x) < .05 && Math.abs(L.y) < .05 && Math.abs(L.vx) < .05 && Math.abs(L.vy) < .05 &&
    Math.abs(L.a) < .002 && Math.abs(L.w) < .002 && L.p < .005);

  function idle(){
    if (held.length || shake > 0.05 || bird.mode !== "perched" || !atRest()) return false;
    if (!active || !cRect) return true;
    const dx = MX - (cRect.left + cRect.width/2), dy = MY - (cRect.top + cRect.height/2);
    const far = P.reach * 3.5;
    return dx*dx + dy*dy > far*far;          // pointer nowhere near: nothing to do
  }

  /* The loop always runs and skips its own work when there is nothing to do — a
     loop that has to be woken by an event stalls for good if a frame is dropped. */
  let last = 0;
  function frame(t){
    requestAnimationFrame(frame);
    let dt = last ? (t - last)/1000 : 0; last = t;
    if (!(dt > 0)) dt = 1/60;
    dt = Math.min(0.05, dt);
    if (idle()) return;

    cRect = host.getBoundingClientRect();
    if (!cRect.width) return;                // brand hidden (narrow viewport)
    for (const L of letters){ L.hx = cRect.left + L.ox; L.hy = cRect.top + L.oy; }

    shake *= Math.exp(-dt*P.shakeDecay);
    if (shake > P.shakeOff && held.length) dropAll();
    tryCatch();

    /* hold on to a letter and the magpie comes to take it back */
    if (held.length){
      holdTime += dt;
      if (bird.mode === "perched" && holdTime > P.patience) launch(held[0]);
    } else if (bird.mode === "perched"){
      holdTime = 0;
    }
    updateBird(dt);
    let rem = dt;
    while (rem > 0){ const h = Math.min(rem, 1/240); step(h); rem -= h; }
    render();
  }

  function init(){
    if (!build()) return;
    measure();
    addEventListener("mousemove", onMove, { passive: true });
    document.addEventListener("mouseleave", () => { active = false; dropAll(); });
    addEventListener("resize", measure);
    addEventListener("scroll", () => { if (idle()) cRect = host.getBoundingClientRect(); }, { passive: true });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    requestAnimationFrame(frame);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
