// frontend/js/facets.js -- populates the sidebar's "Metadata filter"
// field/value dropdowns from GET /api/facets (backend/metadata_filter.py)
// and wires the field -> value cascade. Values come from the server rather
// than being hardcoded here so a newly-extracted lot's facet just shows up
// on next page load, no frontend change needed.

import { getFacets } from "./api.js";

const fieldEl = document.getElementById("facet-field");
const valueEl = document.getElementById("facet-value");

function populateValues(facets, field) {
    valueEl.innerHTML = "";
    if (!field) {
        valueEl.disabled = true;
        valueEl.append(new Option("— pick a field first —", ""));
        return;
    }
    valueEl.disabled = false;
    valueEl.append(new Option("(any)", ""));
    for (const value of facets[field] || []) {
        valueEl.append(new Option(value, value));
    }
}

// `onFieldChange` fires after facet-value's options are already reset for
// the newly-picked field -- callers (app.js) pass their re-search trigger
// here rather than registering their own "change" listener on facet-field,
// so the two can't race (see app.js's comment at the "facet-value" list).
export async function initFacets(onFieldChange) {
    let facets = {};
    try {
        facets = await getFacets();
    } catch (e) {
        // Backend unreachable or no pipeline/*.csv extracted yet -- leave
        // the "(none)" field option in place, filter stays a no-op.
        return;
    }
    for (const field of Object.keys(facets).sort()) {
        fieldEl.append(new Option(field, field));
    }
    fieldEl.addEventListener("change", () => {
        populateValues(facets, fieldEl.value);
        if (onFieldChange) onFieldChange();
    });
}
