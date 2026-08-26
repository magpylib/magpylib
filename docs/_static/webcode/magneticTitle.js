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
 * Move onto a letter and it latches on and becomes the pointer itself, native
 * cursor hidden, until you shake it off. Hold on to it too long and the magpie lifts off the logo, chases the
 * letter down, takes it back and returns it to the wordmark before flying home.
 * The letters always spring back to the baseline, and their travel is capped so
 * the brand stays readable and the link underneath stays easy to hit.
 *
 * The letters only ever move because the reader moved the pointer. The magpie is
 * the one part of this that acts on its own, so it is the part that is kept on a
 * leash: click it and it goes back into the logo for good (click it again and it
 * comes back out), it helps itself to a letter once a visit rather than once a
 * minute, it only does so once the reader has actually stopped, and it stashes
 * what it takes in a nest in the corner rather than on the paragraph being read.
 *
 * Styling lives in _static/custom.css (.mgw / .mgl).
 */
(() => {
  "use strict";

  const WORD = "Magpylib";

  /* Motion is the entire point of this, so there is nothing to degrade to. */
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  /* A touch device has no pointer for any of this to answer to. The wordmark
     still gets its colours, but nothing moves and no magpie flies about a page
     the reader has no way to interact with. */
  const FINE = !window.matchMedia ||
    window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  const P = {
    /* absolute (px/s^2, 1/s) */
    strength: 300, spring: 320, damp: 12, torque: 80,
    rspring: 160, rdamp: 9,
    maxForce: 9000, maxSpring: 18000, maxSpeed: 2600,
    /* how a latched letter is carried */
    /* stiff enough to read as the cursor itself, loose enough to have weight */
    grip: 6000, gripDamp: 155, gripTorque: 420, gripRDamp: 30,
    fling: 0.5, cooldown: 0.55, maxHeld: 1,
    /* the magpie: how long it tolerates a theft, and how it flies */
    patience: 1.8, birdMax: 780, birdSteer: 6,
    /* how far out it starts slowing down: short for a dive, long for a landing */
    slowHunt: 70, slowCarry: 170, slowHome: 220,
    hoverTime: 0.45, dropSpeed: 240, flapBrake: 0.7, landTime: 0.22,
    birdCatch: 26, birdDrop: 12, birdCool: 0.9,
    flapRate: 29,
    /* easing round a turn instead of mirroring on the spot */
    turnRate: 11, turnMin: 0.34,
    /* a shake is counted in direction reversals, not raw speed, so dragging a
       letter quickly across the page keeps it and only a wiggle sheds it */
    shakeSpeed: 380, shakeDecay: 2.2, shakeOff: 2.6,
    /* the horseshoe in the logo is a magnet too */
    logoShare: 0.45, logoPull: 0.06,
    /* the wordmark reads as magnet-coloured even untouched, with headroom left
       for the pointer to drive it the rest of the way */
    baseTint: 0.55,
    /* once the reader has genuinely stopped, the magpie helps itself to a letter */
    stealAfter: 24, stealSpread: 12, stashFor: 22,
    /* derived from the rendered font size in measure() */
    soft: 26, reach: 96, limit: 7, dHalf: 40,
  };

  /* Whether the magpie is out, and whether it has already had its one theft, both
     outlive the page: the first for good, the second for the visit. Storage throws
     outright under some privacy settings, so every access is guarded -- failing
     just means the preference does not stick, never that the script stops. */
  const OFF_KEY = "magpylib-magpie-off", STOLE_KEY = "magpylib-magpie-stole";
  function stored(store, key){
    try { return window[store] && window[store].getItem(key) === "1"; } catch (e){ return false; }
  }
  function remember(store, key, on){
    try { if (window[store]) on ? window[store].setItem(key, "1") : window[store].removeItem(key); } catch (e){}
  }
  let birdOn = !stored("localStorage", OFF_KEY);
  /* Once a visit, not once a page: a reader clicking through ten pages of the
     documentation should meet the joke once, not ten times. */
  let stoleAlready = stored("sessionStorage", STOLE_KEY);

  let host = null, cRect = null, mRect = null, needSettle = true;
  const letters = [], held = [];
  let MX = -1e5, MY = -1e5, active = false;
  let cvx = 0, cvy = 0, lastT = 0, lastX = 0, lastY = 0, shake = 0;

  /* Where the magpie sits inside the logo image, as fractions of it. Kept in step
     with _static/images/magpylib_logo_bird.png, which is that same region cropped
     out, and with ..._nobird.png, which is the logo without it. */
  const BIRD_BOX = { cx: 0.1906, cy: 0.2611, w: 0.3811, h: 0.5222 };
  /* the horseshoe's two pole faces sit side by side at its foot, so its far field
     is a dipole at their midpoint with the moment along the line joining them */
  const LOGO_MAG = { cx: 0.52, cy: 0.86, mx: 1, my: 0 };
  /* The flight frames are a separate drawing: a magpie in a flying posture with
     its proper white markings, six frames of one wing-beat. Sized so its head and
     body match the perched bird's, which is what makes taking off read as the same
     bird rather than a bigger one -- the box is wider only because the wings are
     open. Measured from _static/images/magpylib_logo_bird_fly.png. */
  const FLY = { w: 1.288, h: 1.202, clawX: 0.772, clawY: 0.978 };
  /* Six drawings, ordered from wings-highest to wings-lowest. Playing them out and
     back gives a ten-step beat without drawing the upstroke twice. */
  const FLAP = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1];
  /* The perched bird carries the logo's own outline, baked in at 1.573% of the
     logo's width. The flight frames are drawn without one, so CSS strokes it --
     at the same fraction, or the two versions of the bird do not match. */
  const RIM = 0.01573;
  const bird = { el:null, body:null, still:null, eye:null, mode:"perched", target:null, errand:"recover",
                 x:0, y:0, vx:0, vy:0, w:0, h:0, t:0, ph:0, face:1, turn:1, hold:0, lt:0, fromW:0, fromH:0, fromX:0, fromY:0, stashX:0, stashY:0 };
  let holdTime = 0, logoImgs = [], srcPerched = [], srcFlown = [];
  let idleTime = 0, stealAt = 0, stashTimer = 0, perchWatch = 0;
  let wearingSwat = false;

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
                     x:0, y:0, vx:0, vy:0, a:0, w:0, p:0, pPrev:0,
                     ax:0, ay:0,            // where the spring pulls it back to
                     stuck:false, carried:false, stashed:false, cool:0 });
    }
    host.textContent = "";
    host.appendChild(wrap);

    if (birdOn) makeBird();

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
  /* Built on demand rather than once, because the reader can send it away and ask
     for it back within the same page. */
  function makeBird(){
    if (bird.el) return;
    bird.el = document.createElement("div");
    bird.el.className = "mag-bird";
    bird.el.setAttribute("aria-hidden", "true");
    /* The logo magpie is one flat silhouette with no wing to animate, so the wing
       is a separate shape in the same ink, hinged at the shoulder. Both sit under
       one drop-shadow, so they read as a single beating silhouette. */
    bird.body = document.createElement("div");
    bird.body.className = "mag-bird__body";
    /* the perched drawing lives on its own layer, so landing can fade from one
       to the other instead of swapping sprites in a single frame */
    bird.still = document.createElement("div");
    bird.still.className = "mag-bird__still";
    bird.eye = document.createElement("div");
    bird.eye.className = "mag-bird__eye";
    bird.still.appendChild(bird.eye);
    bird.el.appendChild(bird.body);
    bird.el.appendChild(bird.still);
    document.body.appendChild(bird.el);
  }

  function logoRect(){
    /* the theme ships a light and a dark logo and hides one of them */
    for (const img of logoImgs){
      const r = img.getBoundingClientRect();
      if (r.width) return r;
    }
    return null;
  }

  function perch(){
    const r = logoRect();
    if (!r) return null;
    return { x: r.left + BIRD_BOX.cx * r.width, y: r.top + BIRD_BOX.cy * r.height,
             w: BIRD_BOX.w * r.width, h: BIRD_BOX.h * r.height };
  }

  function flyBox(){
    const r = logoRect();
    if (!r) return null;
    bird.el.style.setProperty("--rim", (RIM * r.width).toFixed(2) + "px");
    return { w: FLY.w * r.width, h: FLY.h * r.width };
  }

  /* where the claws are, relative to the bird's centre, mirrored when it banks */
  function clawDX(){ return (FLY.clawX - 0.5) * bird.w * (bird.turn < 0 ? -1 : 1); }
  function clawDY(){ return (FLY.clawY - 0.5) * bird.h; }

  function showLogoBird(on){
    if (!srcFlown.length) return;
    logoImgs.forEach((img, i) => img.setAttribute("src", on ? srcPerched[i] : srcFlown[i]));
  }

  /* Put the live magpie on the perch and take the painted one out of the logo.
     Done once the perch is known to be measurable, so a logo that never resolves
     keeps its own bird rather than losing it to an element that never appeared. */
  function settle(){
    if (!birdOn || !bird.el) return false;
    const p = perch();
    if (!p){
      /* no measurable logo (narrow viewport, hidden brand): give the painted
         bird back rather than leaving one floating and the other missing */
      bird.el.style.opacity = "0";
      bird.el.classList.remove("mag-bird--perched");
      showLogoBird(true);
      return false;
    }
    bird.x = p.x; bird.y = p.y; bird.w = p.w; bird.h = p.h;
    bird.vx = 0; bird.vy = 0; bird.face = 1;
    bird.el.style.width = p.w.toFixed(1) + "px";
    bird.el.style.height = p.h.toFixed(1) + "px";
    bird.el.style.transform =
      "translate3d(" + (p.x - p.w/2).toFixed(1) + "px," + (p.y - p.h/2).toFixed(1) + "px,0)";
    bird.el.style.opacity = "1";
    bird.el.classList.remove("mag-bird--flying");
    bird.el.classList.add("mag-bird--perched");
    bird.body.style.opacity = "0";
    bird.still.style.opacity = "1";
    bird.turn = bird.face = 1;
    bird.body.style.transform = "";
    showLogoBird(false);
    return true;
  }

  function launch(target, errand){
    if (!birdOn || !bird.el) return;
    const p = perch();
    if (!p) return;
    bird.errand = errand;
    const f = flyBox() || { w: p.w, h: p.h };
    bird.x = p.x; bird.y = p.y; bird.w = f.w; bird.h = f.h;
    bird.vx = 0; bird.vy = 0; bird.t = 0; bird.ph = 0;
    bird.mode = "hunt"; bird.target = target;
    bird.el.style.width = f.w.toFixed(1) + "px";
    bird.el.style.height = f.h.toFixed(1) + "px";
    bird.el.style.opacity = "1";
    bird.el.classList.remove("mag-bird--perched");
    bird.el.classList.add("mag-bird--flying");
    bird.body.style.opacity = "1";
    bird.still.style.opacity = "0";
  }

  function land(){
    const p = perch();
    if (!p){ bird.mode = "perched"; bird.target = null; settle(); return; }
    /* ease down onto the perch: the flying drawing fades out as the perched one
       fades in, and the box shrinks to match, so there is no jump */
    bird.mode = "landing"; bird.target = null; bird.lt = 0;
    bird.fromW = bird.w; bird.fromH = bird.h;
    bird.fromX = bird.x; bird.fromY = bird.y;
  }

  /* Steering with an arrival: the speed it *wants* tapers to nothing inside the
     slowing radius, so it settles onto the target. Accelerating flat out until it
     gets there is what made it shoot past the perch and circle back round. */
  function seek(tx, ty, dt, slow){
    const dx = tx - bird.x, dy = ty - bird.y;
    const d = Math.hypot(dx, dy) || 1e-6;
    const want = P.birdMax * Math.min(1, d / (slow || P.slowHome));
    const k = Math.min(1, dt * P.birdSteer);
    bird.vx += (dx/d * want - bird.vx) * k;
    bird.vy += (dy/d * want - bird.vy) * k;
    bird.x += bird.vx*dt; bird.y += bird.vy*dt;
    return d;
  }

  function updateBird(dt){
    if (bird.mode === "perched") return;
    bird.t += dt;

    if (bird.mode === "hunt"){
      const L = bird.target;
      /* the errand can be called off underneath it: shaken loose before it
         arrived, or the reader picked the stashed letter up first */
      const gone = !L || L.carried ||
        (bird.errand === "recover" && !L.stuck) ||
        (bird.errand === "fetch" && !L.stashed);
      if (gone){
        bird.mode = "home"; bird.target = null;
      } else if (seek(L.hx + L.x, L.hy + L.y, dt, P.slowHunt) < P.birdCatch){
        const i = held.indexOf(L);
        if (i >= 0) held.splice(i, 1);
        L.stuck = false; L.stashed = false; L.carried = true; L.cool = Infinity;
        holdTime = 0; shake = 0; stashTimer = 0;
        bird.mode = "carry";
      }
    }
    if (bird.mode === "carry"){
      const L = bird.target;
      const stealing = bird.errand === "steal";
      const tx = (stealing ? bird.stashX : L.hx) - clawDX();
      const ty = (stealing ? bird.stashY : L.hy) - clawDY();
      if (seek(tx, ty, dt, P.slowCarry) < P.birdDrop){
        L.carried = false; L.cool = P.birdCool;      // a moment before it can be taken again
        L.vy += P.dropSpeed;                         // it falls the last little way
        if (stealing){
          L.stashed = true;                          // the spring now holds it here
          L.ax = bird.stashX - L.hx; L.ay = bird.stashY - L.hy;
          stashTimer = P.stashFor;
        } else {
          L.ax = 0; L.ay = 0;
        }
        bird.hold = P.hoverTime;
        bird.mode = "deliver";
      }
    }
    if (bird.mode === "deliver"){
      /* hang over it while it drops into place, instead of turning tail in the
         same frame it lets go */
      const L = bird.target;
      const tx = (bird.errand === "steal" ? bird.stashX : L.hx) - clawDX();
      const ty = (bird.errand === "steal" ? bird.stashY : L.hy) - clawDY() - bird.h*0.12;
      seek(tx, ty, dt, 40);
      bird.hold -= dt;
      if (bird.hold <= 0){ bird.target = null; bird.mode = "home"; }
    }
    if (bird.mode === "home"){
      const p = perch();
      if (!p){ land(); return; }
      if (seek(p.x, p.y, dt, P.slowHome) < 4){ land(); return; }
    }
    if (bird.mode === "landing"){
      const p = perch();
      if (!p){ bird.mode = "perched"; settle(); return; }
      bird.lt += dt;
      const k = Math.min(1, bird.lt / P.landTime);
      const e = k*k*(3 - 2*k);                       // smoothstep
      bird.x = bird.fromX + (p.x - bird.fromX)*e;
      bird.y = bird.fromY + (p.y - bird.fromY)*e;
      bird.w = bird.fromW + (p.w - bird.fromW)*e;
      bird.h = bird.fromH + (p.h - bird.fromH)*e;
      bird.vx *= 0.85; bird.vy *= 0.85;
      bird.el.style.width  = bird.w.toFixed(1) + "px";
      bird.el.style.height = bird.h.toFixed(1) + "px";
      bird.body.style.opacity  = (1-e).toFixed(3);
      bird.still.style.opacity = e.toFixed(3);
      if (k >= 1){ bird.mode = "perched"; settle(); return; }
    }

    /* One flap drives both wings and the body. The far wing lags, so at any
       instant the two are at visibly different angles; without that they overlap
       into a single lump. Both shorten at the ends of the stroke, where a real
       wing is swinging towards the viewer rather than across it. */
    /* it flutters faster the slower it is going, the way a bird brakes onto a perch */
    const slowness = Math.max(0, 1 - Math.hypot(bird.vx, bird.vy) / P.birdMax);
    bird.ph += dt * P.flapRate * (1 + P.flapBrake * slowness);
    /* the wing-beat is drawn, so flapping is just which frame is showing */
    const n = FLAP.length;
    const step = ((Math.floor(bird.ph / (Math.PI*2) * n) % n) + n) % n;
    bird.body.style.setProperty("--frame", FLAP[step]);

    const bob = -Math.sin(bird.ph) * bird.h * 0.03;
    if (Math.abs(bird.vx) > 40) bird.face = bird.vx > 0 ? -1 : 1;   // the sprite faces left

    /* Turning is eased rather than mirrored on the spot: the sprite narrows as it
       comes edge-on and widens out the other side, which reads as swinging round
       in depth instead of flipping. It never goes fully edge-on, or it vanishes. */
    bird.turn += (bird.face - bird.turn) * Math.min(1, dt * P.turnRate);
    const across = Math.max(P.turnMin, Math.abs(bird.turn)) * (bird.turn < 0 ? -1 : 1);

    /* the drawing already holds a flying posture, so this is only the pitch */
    const tilt = Math.max(-0.4, Math.min(0.4, bird.vy / 1100)) * bird.face;
    bird.el.style.transform =
      "translate3d(" + (bird.x - bird.w/2).toFixed(1) + "px," + (bird.y - bird.h/2 + bob).toFixed(1) + "px,0)" +
      " scale(" + across.toFixed(3) + ",1)" +
      " rotate(" + tilt.toFixed(3) + "rad)";
  }

  /* Letter homes are kept as offsets inside the brand, so the sticky header can
     move without re-measuring every glyph. */
  function measure(){
    for (const L of letters) L.el.style.transform = "none";
    cRect = host.getBoundingClientRect();
    if (bird.mode === "perched") settle();
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
    /* the nest is placed off the viewport, so a resize moves it: re-aim anything
       already on its way there, and re-anchor anything already sitting in it */
    const n = nest();
    if (bird.errand === "steal" && bird.mode !== "perched"){ bird.stashX = n.x; bird.stashY = n.y; }
    for (const L of letters){
      if (!L.stashed) continue;
      L.ax = n.x - (cRect.left + L.ox);
      L.ay = n.y - (cRect.top  + L.oy);
    }
    mRect = logoRect();
    needSettle = true;
    render();
  }

  /* ---------- sending the magpie away, and asking it back ---------- */
  /* The bird element has pointer-events: none so that it can never swallow a click
     meant for the page under it. The hit is tested here instead, against a tight
     ellipse on its body, which is also what stops the perched bird from taking
     clicks on the brand link it is sitting on -- the wordmark and the rest of the
     logo stay live. While it is away the same spot on the painted logo answers,
     so the gesture is a toggle rather than a one-way door: nobody loses the bird
     for good by clicking where they meant to click "home". */
  function birdBox(){
    if (birdOn && bird.el && bird.mode !== "perched")
      return { x: bird.x, y: bird.y, w: bird.w, h: bird.h, flying: true };
    const p = perch();
    return p ? { x: p.x, y: p.y, w: p.w, h: p.h, flying: false } : null;
  }

  function hitsBird(x, y){
    const b = birdBox();
    if (!b) return false;
    const rx = b.w * (b.flying ? 0.26 : 0.34), ry = b.h * (b.flying ? 0.30 : 0.40);
    const dx = (x - b.x)/rx, dy = (y - b.y)/ry;
    return dx*dx + dy*dy <= 1;
  }

  function dismiss(){
    birdOn = false;
    remember("localStorage", OFF_KEY, true);
    if (held.length) dropAll();
    /* whatever it was carrying goes back to the wordmark, rather than being left
       wherever it happened to be at the moment it was swatted */
    for (const L of letters){
      L.carried = false; L.stashed = false; L.ax = 0; L.ay = 0;
      if (L.cool === Infinity) L.cool = P.cooldown;    // clear the carried hold
    }
    stashTimer = 0; idleTime = 0;
    bird.mode = "perched"; bird.target = null;
    if (bird.el){ bird.el.remove(); bird.el = null; }
    showLogoBird(true);            // the painted magpie comes back, and stays put
    needSettle = true;
  }

  function revive(){
    birdOn = true;
    remember("localStorage", OFF_KEY, false);
    bird.mode = "perched"; bird.target = null;
    idleTime = 0;
    stealAt = P.stealAfter + Math.random()*P.stealSpread;
    makeBird();
    settle();
  }

  /* ---------- pointer ---------- */
  /* Any sign of life resets the boredom clock. Without this the magpie treated a
     reader working their way down a long page as an empty room, and went thieving
     over the top of what they were reading. */
  function stir(){ idleTime = 0; }

  function onMove(e){
    MX = e.clientX; MY = e.clientY; active = true;
    stir();

    /* the bird cannot carry a cursor of its own, so the hint goes on the root */
    const swat = !held.length && hitsBird(MX, MY);
    if (swat !== wearingSwat){
      wearingSwat = swat;
      document.documentElement.classList.toggle("mag-swat", swat);
    }

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
      if (Math.abs(lx) < L.hw && Math.abs(ly) < L.hh){
        L.stuck = true; held.push(L);
        /* taking a stashed letter back by hand cancels the magpie's errand */
        if (L.stashed){ L.stashed = false; L.ax = 0; L.ay = 0; stashTimer = 0; }
      }
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

  /* ---------- field of the logo's own horseshoe ----------
     The letters are soft magnetic: they carry no moment of their own, they are
     magnetised by whatever field reaches them. So the logo tints the letters
     nearest it and tugs them very slightly towards itself (|B| rises towards a
     magnet, and a soft body is drawn up that gradient) but imposes no preferred
     angle -- which is what keeps the wordmark sitting straight when untouched. */
  function logoMag(px, py, C){
    if (!mRect) return 0;
    const sx = mRect.left + LOGO_MAG.cx * mRect.width;
    const sy = mRect.top  + LOGO_MAG.cy * mRect.height;
    const rx = px - sx, ry = py - sy;
    const d2 = rx*rx + ry*ry + P.soft*P.soft;
    const d = Math.sqrt(d2);
    const md = (LOGO_MAG.mx*rx + LOGO_MAG.my*ry) / d;
    const k = C/(d2*d);
    const bx = k*(3*md*rx/d - LOGO_MAG.mx);
    const by = k*(3*md*ry/d - LOGO_MAG.my);
    return Math.hypot(bx, by);
  }

  function step(dt){
    const C  = P.strength * Math.pow(P.reach, 4) / 3;
    const B0 = C / (P.dHalf*P.dHalf*P.dHalf);   // sets how close is "fully magnetised"
    const e = 1;

    for (const L of letters){
      if (L.cool > 0) L.cool -= dt;

      if (L.stuck || L.carried){
        let tx, ty;
        if (L.carried){
          tx = bird.x + clawDX(); ty = bird.y + clawDY();   // gripped in the claws
        } else {
          tx = MX; ty = MY;                          // the letter *is* the pointer now
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

      const px0 = L.hx + L.x, py0 = L.hy + L.y;
      let fx = 0, fy = 0, tq = 0, pol = 0;

      /* the logo's magnet: a tint, and a whisper of a pull */
      const Cl = C * P.logoShare;
      const bl = logoMag(px0, py0, Cl);
      if (bl > 0){
        fx += P.logoPull * (logoMag(px0+e, py0, Cl) - logoMag(px0-e, py0, Cl)) / (2*e);
        fy += P.logoPull * (logoMag(px0, py0+e, Cl) - logoMag(px0, py0-e, Cl)) / (2*e);
      }

      if (active){
        const px = px0, py = py0;
        const ca = Math.cos(L.a), sa = Math.sin(L.a);
        const lmx = sa, lmy = -ca;                        // this letter's moment
        fieldAt(px+e, py, C); const a1 = lmx*BX + lmy*BY;
        fieldAt(px-e, py, C); const a2 = lmx*BX + lmy*BY;
        fieldAt(px, py+e, C); const a3 = lmx*BX + lmy*BY;
        fieldAt(px, py-e, C); const a4 = lmx*BX + lmy*BY;
        fx += (a1-a2)/(2*e); fy += (a3-a4)/(2*e);          // F = grad(m.B)
        fieldAt(px, py, C);
        const bm = Math.hypot(BX, BY) || 1e-9;
        tq  = P.torque * ((ca*BX + sa*BY)/bm) * (bm/(bm+B0));   // aligns m with B
      }
      const fm = Math.hypot(fx, fy);
      if (fm > P.maxForce){ fx *= P.maxForce/fm; fy *= P.maxForce/fm; }
      {
        fieldAt(px0, py0, C);
        const bm = active ? Math.hypot(BX, BY) : 0;
        pol = (bm + bl) / (bm + bl + B0);
      }

      /* The restoring spring stiffens with displacement, so a letter can be pulled
         hard but never far enough to scramble the brand or shift the link target.
         Both it and the speed are capped: a letter dropped across the page would
         otherwise make this spring explode. */
      const rx = L.x - L.ax, ry = L.y - L.ay;      // displacement from its anchor
      const off = Math.hypot(rx, ry) / P.limit;
      const stiff = 1 + Math.min(off*off, 36);
      let sx = -P.spring*rx*stiff, sy = -P.spring*ry*stiff;
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

  let wearingLetter = false;
  function render(){
    /* Driven from the rendered state rather than from the events that change it,
       so the cursor can never be left hidden by a path that forgot to restore it. */
    const carrying = held.length > 0;
    if (carrying !== wearingLetter){
      wearingLetter = carrying;
      document.documentElement.classList.toggle("mag-carrying", carrying);
    }
    for (const L of letters){
      L.el.style.transform = "translate3d(" + L.x.toFixed(2) + "px," + L.y.toFixed(2) + "px,0) rotate(" + L.a.toFixed(4) + "rad)";
      L.el.style.setProperty("--p", (P.baseTint + (1 - P.baseTint) * L.p).toFixed(3));
    }
  }

  /* The logo keeps the nearest letters faintly magnetised for ever, so "at rest"
     cannot mean "back at zero" -- it means nothing is changing any more. */
  const atRest = () => letters.every(L =>
    Math.abs(L.vx) < .05 && Math.abs(L.vy) < .05 && Math.abs(L.w) < .002 &&
    Math.abs(L.p - L.pPrev) < .0008);

  function idle(){
    if (needSettle || held.length || shake > 0.05 || bird.mode !== "perched") return false;
    if (letters.some(L => L.stashed) || !atRest()) return false;
    if (!active || !cRect) return true;
    const dx = MX - (cRect.left + cRect.width/2), dy = MY - (cRect.top + cRect.height/2);
    const far = P.reach * 3.5;
    return dx*dx + dy*dy > far*far;          // pointer nowhere near: nothing to do
  }

  /* ---------- left alone, the magpie helps itself ---------- */
  /* Where a stolen letter gets put down. It used to be anywhere across the middle
     of the viewport, which is exactly where the text being read is -- so the
     letter landed on the paragraph. The nest is one fixed spot in the bottom
     corner instead: out of the column, and the same place every time, so it can
     be spotted and taken back rather than hunted for.
     Bottom right specifically: the theme parks its own back-to-top button at
     bottom centre (left: 50vw, top: 90vh), and the letter is kept off the very
     edge so that dropping it cannot push the page's scrollable area out. */
  function nest(){
    return { x: innerWidth  - Math.max(56, innerWidth  * 0.06),
             y: innerHeight - Math.max(72, innerHeight * 0.14) };
  }

  function startSteal(){
    if (!birdOn || stoleAlready) return;
    const pool = letters.filter(L => !L.stuck && !L.carried && !L.stashed && L.cool <= 0);
    if (!pool.length) return;
    const n = nest();
    bird.stashX = n.x; bird.stashY = n.y;
    launch(pool[(Math.random()*pool.length)|0], "steal");
    if (bird.mode !== "perched"){         // it actually got off the ground
      stoleAlready = true;
      remember("sessionStorage", STOLE_KEY, true);
    }
    idleTime = 0;
    stealAt = P.stealAfter + Math.random()*P.stealSpread;
  }

  /* The loop always runs and skips its own work when there is nothing to do — a
     loop that has to be woken by an event stalls for good if a frame is dropped. */
  let last = 0;
  function frame(t){
    requestAnimationFrame(frame);
    let dt = last ? (t - last)/1000 : 0; last = t;
    if (!(dt > 0)) dt = 1/60;
    stepFrame(Math.min(0.05, dt));
  }

  function stepFrame(dt){
    /* The perched bird is placed in viewport coordinates, so anything that moves
       the header under it -- a version banner arriving, a webfont landing, the
       sidebar opening -- strands it where the logo used to be. None of those fire
       resize, so watch the perch rather than measuring it once. */
    if (birdOn && bird.el && bird.mode === "perched"){
      perchWatch += dt;
      if (perchWatch > 0.4){
        perchWatch = 0;
        const p = perch();
        if (!p || Math.abs(p.x - bird.x) > 0.5 || Math.abs(p.y - bird.y) > 0.5){
          cRect = host.getBoundingClientRect();
          settle();
        }
      }
    }

    /* This bookkeeping has to run even while the simulation is asleep, otherwise
       the magpie could never get bored enough to go thieving. */
    const stashed = letters.some(L => L.stashed);
    /* "Quiet" now means the reader has stopped, not merely that the magpie is
       sitting still: idleTime is reset by any pointer, scroll or key activity
       (see stir), so reading a long page no longer counts as an empty room. */
    const quiet = birdOn && !stoleAlready && !held.length &&
                  bird.mode === "perched" && !stashed && !document.hidden;
    if (quiet) {
      idleTime += dt;
      /* the longer it waits, the more it fidgets -- so the theft is telegraphed
         rather than coming out of nowhere. Written only when the mood changes. */
      const t = idleTime / (stealAt || 1);
      const mood = t > 0.85 ? "twitchy" : t > 0.5 ? "restless" : "calm";
      if (bird.el && bird.el.dataset.mood !== mood) bird.el.dataset.mood = mood;
      if (idleTime > stealAt) startSteal();
    } else if (!stashed) {
      idleTime = 0;
      if (bird.el && bird.el.dataset.mood !== "calm") bird.el.dataset.mood = "calm";
    }
    /* and it always brings a stolen letter back in the end */
    if (stashTimer > 0){
      stashTimer -= dt;
      if (stashTimer <= 0 && bird.mode === "perched"){
        const L = letters.find(x => x.stashed);
        if (L) launch(L, "fetch"); else stashTimer = 0;
      }
    }

    if (idle()) return;

    cRect = host.getBoundingClientRect();
    if (!cRect.width) return;                // brand hidden (narrow viewport)
    for (const L of letters){ L.hx = cRect.left + L.ox; L.hy = cRect.top + L.oy; L.pPrev = L.p; }
    mRect = logoRect();

    shake *= Math.exp(-dt*P.shakeDecay);
    if (shake > P.shakeOff && held.length) dropAll();
    tryCatch();

    /* hold on to a letter and the magpie comes to take it back */
    if (held.length){
      holdTime += dt;
      if (bird.mode === "perched" && holdTime > P.patience) launch(held[0], "recover");
    } else if (bird.mode === "perched"){
      holdTime = 0;
    }
    updateBird(dt);
    let rem = dt;
    while (rem > 0){ const h = Math.min(rem, 1/240); step(h); rem -= h; }
    render();
    if (needSettle && atRest()) needSettle = false;
  }

  function init(){
    if (!build()) return;
    if (!FINE){
      for (const L of letters) L.el.style.setProperty("--p", P.baseTint.toFixed(3));
      if (bird.el){ bird.el.remove(); bird.el = null; }   // logo keeps its painted magpie
      return;
    }
    measure();
    addEventListener("mousemove", onMove, { passive: true });
    const letGo = () => { active = false; dropAll(); render(); };
    document.addEventListener("mouseleave", letGo);
    /* Switching away does not fire mouseleave, and the cursor is hidden while a
       letter is riding on it -- so coming back to an invisible cursor holding a
       letter is the one way this can strand someone. */
    addEventListener("blur", letGo);
    document.addEventListener("visibilitychange", () => { if (document.hidden) letGo(); });
    addEventListener("keydown", stir, { passive: true });
    /* One click on the magpie sends it back into the logo for good; one on the
       same spot brings it out again. Taken in the capture phase and only when the
       pointer is really on the bird, so every other click on the brand link --
       including the one that means "take me home" -- is left alone. */
    addEventListener("click", (e) => {
      if (e.button || !hitsBird(e.clientX, e.clientY)) return;
      e.preventDefault();
      e.stopPropagation();
      if (wearingSwat){ wearingSwat = false; document.documentElement.classList.remove("mag-swat"); }
      birdOn ? dismiss() : revive();
    }, true);
    addEventListener("resize", measure);
    addEventListener("scroll", () => {
      stir();
      if (!idle()) return;
      cRect = host.getBoundingClientRect();
      if (bird.mode === "perched") settle();
    }, { passive: true });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
    settle();
    stealAt = P.stealAfter + Math.random()*P.stealSpread;
    requestAnimationFrame(frame);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
