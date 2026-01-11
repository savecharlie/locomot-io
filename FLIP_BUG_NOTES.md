# Flip Bug Investigation Notes

## Problem
Flip doesn't work - content appears to revert immediately

## History (15+ commits tried)
1. Canvas transforms (scale(-1,1)) - didn't persist
2. ImageData pixel manipulation - same issue
3. Flip to temp canvas then draw - same
4. Read from composite vs layer directly - tried both
5. Clear all layers before write - no help
6. Add autoSave after flip - didn't help
7. Commit floating selection after flip - FIXED floating case
8. Various logging to debug

## Current Code
Two paths in flipSelection():
- **FLOATING**: Flips floatingData, commits to layer (works?)
- **NON-FLOATING**: Reads from layer, creates flipped ImageData, writes back to layer

## ROOT CAUSE FOUND (likely)
Selection toolbar has BOTH click AND touchend handlers (lines 3645-3661):
```javascript
// click handler
selectionToolbar.addEventListener('click', (e) => {
    selectionAction(btn.dataset.action);
});
// touchend handler
selectionToolbar.addEventListener('touchend', (e) => {
    selectionAction(btn.dataset.action);
});
```

On mobile, a tap fires BOTH events:
1. touchend → flip happens (correct)
2. click → flip happens AGAIN (flips back to original!)

Even with preventDefault() in touchend, some browsers still fire click.

## Fix
Remove duplicate handler OR add debounce flag:

Option A - Remove click handler, touchend works for both:
```javascript
// touchend handles mobile taps AND mouse clicks generate touchend
```

Option B - Add flag to prevent double-execution:
```javascript
let lastActionTime = 0;
function selectionAction(action) {
    const now = Date.now();
    if (now - lastActionTime < 100) return; // debounce
    lastActionTime = now;
    // ... rest of function
}
```

## Testing needed
1. Verify double-fire with visible toast (should see 2 toasts per tap)
2. Test fix on mobile
