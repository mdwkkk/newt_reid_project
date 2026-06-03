(function () {
  const $ = (id) => document.getElementById(id);
  const LOW_CONFIDENCE_THRESHOLD = 0.7;
  const THEME_STORAGE_KEY = 'triton-reid-theme';

  /* ---------- тема (светлая / тёмная) ---------- */
  function getTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'light' ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    const next = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem(THEME_STORAGE_KEY, next);
    const btn = $('btnThemeToggle');
    if (btn) {
      const label = next === 'light' ? 'Переключить на тёмную тему' : 'Переключить на светлую тему';
      btn.setAttribute('aria-label', label);
      btn.setAttribute('title', next === 'light' ? 'Тёмная тема' : 'Светлая тема');
    }
  }

  function initThemeToggle() {
    const btn = $('btnThemeToggle');
    if (!btn) return;
    applyTheme(getTheme());
    btn.addEventListener('click', () => {
      applyTheme(getTheme() === 'dark' ? 'light' : 'dark');
    });
  }

  /* ---------- глобальное боковое меню ---------- */
  function initNavDrawer() {
    const drawer = $('navDrawer');
    const overlay = $('navOverlay');
    const btnOpen = $('btnNavMenu');
    const btnClose = $('btnNavClose');
    if (!drawer || !btnOpen) return;

    function setOpen(open) {
      drawer.classList.toggle('is-open', open);
      overlay?.classList.toggle('hidden', !open);
      btnOpen.setAttribute('aria-expanded', open ? 'true' : 'false');
      drawer.setAttribute('aria-hidden', open ? 'false' : 'true');
      document.body.style.overflow = open ? 'hidden' : '';
    }

    btnOpen.addEventListener('click', () => setOpen(true));
    btnClose?.addEventListener('click', () => setOpen(false));
    overlay?.addEventListener('click', () => setOpen(false));
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && drawer.classList.contains('is-open')) setOpen(false);
    });
    drawer.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', () => setOpen(false));
    });
  }

  function initGlobalUi() {
    initNavDrawer();
    initThemeToggle();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initGlobalUi);
  } else {
    initGlobalUi();
  }

  function showToast(msg, isError) {
    let el = $('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.className = 'toast';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.className = 'toast' + (isError ? ' is-error' : '');
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 4000);
  }

  async function apiJson(url, opts) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => d.msg || d).join('; ') : res.statusText;
      throw new Error(msg || 'Ошибка запроса');
    }
    return data;
  }

  function renderMetadataFields(container, schema, metadata, opts) {
    const { readonly = false, projectId = null, allIds = [] } = opts || {};
    container.replaceChildren();
    (schema.fields || []).forEach((field) => {
      const wrap = document.createElement('div');
      if (field.type === 'textarea') wrap.className = 'span-2';
      const label = document.createElement('label');
      label.className = 'label';
      label.textContent = field.label_ru + (field.required ? ' *' : '');
      label.htmlFor = 'f_' + field.key;

      let input;
      const val = metadata[field.key] ?? '';
      if (readonly) {
        const dt = document.createElement('div');
        dt.className = 'text-slate-200';
        if (field.type === 'individual_ref' && val && projectId) {
          const a = document.createElement('a');
          a.href = '/projects/' + projectId + '/individuals/' + encodeURIComponent(val);
          a.className = 'field-ref-link';
          a.textContent = val;
          dt.appendChild(a);
        } else {
          dt.textContent = val || '—';
        }
        wrap.appendChild(label);
        wrap.appendChild(dt);
      } else if (field.type === 'textarea') {
        input = document.createElement('textarea');
        input.className = 'input';
        input.value = val;
      } else if (field.type === 'select') {
        input = document.createElement('select');
        input.className = 'input';
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '—';
        input.appendChild(empty);
        (field.options || []).forEach((o) => {
          const opt = document.createElement('option');
          opt.value = o;
          opt.textContent = o;
          if (o === val) opt.selected = true;
          input.appendChild(opt);
        });
      } else if (field.type === 'individual_ref') {
        input = document.createElement('input');
        input.type = 'text';
        input.className = 'input';
        input.style.fontFamily = 'var(--font-mono)';
        input.value = val;
        input.setAttribute('list', 'individual-ids-list');
        input.placeholder = 'ID из проекта';
      } else {
        input = document.createElement('input');
        input.type = field.type === 'number' ? 'number' : field.type === 'date' ? 'date' : field.type === 'time' ? 'time' : 'text';
        input.className = 'input';
        input.value = val;
      }
      if (!readonly && input) {
        input.id = 'f_' + field.key;
        input.name = field.key;
        if (field.key === 'individual_id' && window.MODE === 'create') {
          input.required = true;
        }
        wrap.appendChild(label);
        wrap.appendChild(input);
      }
      container.appendChild(wrap);
    });
    if (!readonly && allIds.length) {
      let dl = document.getElementById('individual-ids-list');
      if (!dl) {
        dl = document.createElement('datalist');
        dl.id = 'individual-ids-list';
        document.body.appendChild(dl);
      }
      dl.replaceChildren();
      allIds.forEach((id) => {
        const opt = document.createElement('option');
        opt.value = id;
        dl.appendChild(opt);
      });
    }
  }

  function collectMetadata(container, schema) {
    const meta = {};
    (schema.fields || []).forEach((field) => {
      const el = $('f_' + field.key);
      if (el) meta[field.key] = el.value.trim();
    });
    return meta;
  }

  /* ---------- projects page ---------- */
  if (window.PAGE === 'projects') {
    const modal = $('createModal');
    const form = $('createForm');
    const sel = $('cardTemplate');
    const desc = $('templateDesc');
    let templates = [];

    function openCreateModal() {
      modal?.classList.remove('hidden');
    }
    $('btnOpenCreate')?.addEventListener('click', openCreateModal);
    $('btnOpenCreateEmpty')?.addEventListener('click', openCreateModal);
    document.querySelectorAll('[data-close-modal]').forEach((el) => {
      el.addEventListener('click', () => modal?.classList.add('hidden'));
    });

    apiJson('/api/card-templates').then((data) => {
      templates = data.templates || [];
      templates.forEach((t) => {
        const o = document.createElement('option');
        o.value = t.code;
        o.textContent = t.name_ru + ' (' + t.code + ')';
        sel?.appendChild(o);
      });
      updateDesc();
    });

    function updateDesc() {
      const code = sel?.value;
      apiJson('/api/card-templates/' + code).then((tpl) => {
        if (desc) desc.textContent = tpl.description || '';
      }).catch(() => {});
    }
    sel?.addEventListener('change', updateDesc);

    form?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = $('projectName')?.value?.trim();
      const card_template = sel?.value;
      try {
        const proj = await apiJson('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, card_template }),
        });
        window.location.href = '/projects/' + proj.id;
      } catch (err) {
        showToast(err.message, true);
      }
    });

  }

  /* ---------- search page ---------- */
  if (window.PAGE === 'search' && window.PROJECT_ID) {
    const pid = window.PROJECT_ID;
    const fileInput = $('file');
    const btnPick = $('btnPick');
    const btnRun = $('btnRun');
    const preview = $('preview');
    const previewEmpty = $('previewEmpty');
    const uploadZone = $('uploadZone');
    const spinner = $('spinner');
    const results = $('results');
    const alertEl = $('alert');
    const emptyBox = $('emptyProject');
    const lowConfBanner = $('lowConfidenceBanner');
    const btnDismissLowConf = $('btnDismissLowConf');
    const btnShowLowConfHint = $('btnShowLowConfHint');
    let fileObj = null;
    let lowConfDismissed = false;

    function hideLowConfidenceUi() {
      lowConfBanner?.classList.add('hidden');
      btnShowLowConfHint?.classList.add('hidden');
      lowConfDismissed = false;
    }

    function updateLowConfidenceUi(confidence) {
      const conf = confidence != null ? Number(confidence) : null;
      const isLow = conf != null && conf < LOW_CONFIDENCE_THRESHOLD;
      if (!isLow) {
        hideLowConfidenceUi();
        return;
      }
      if (!lowConfDismissed) {
        lowConfBanner?.classList.remove('hidden');
        btnShowLowConfHint?.classList.add('hidden');
      } else {
        lowConfBanner?.classList.add('hidden');
        btnShowLowConfHint?.classList.remove('hidden');
      }
    }

    btnDismissLowConf?.addEventListener('click', () => {
      lowConfDismissed = true;
      lowConfBanner?.classList.add('hidden');
      btnShowLowConfHint?.classList.remove('hidden');
    });

    btnShowLowConfHint?.addEventListener('click', () => {
      lowConfDismissed = false;
      lowConfBanner?.classList.remove('hidden');
      btnShowLowConfHint?.classList.add('hidden');
    });

    function setSpinner(on) {
      spinner?.classList.toggle('is-visible', !!on);
      if (btnRun) btnRun.disabled = on || !fileObj;
    }

    function showError(msg) {
      if (!alertEl) return;
      if (!msg) {
        alertEl.classList.add('hidden');
        return;
      }
      alertEl.textContent = msg;
      alertEl.classList.remove('hidden');
    }

    btnPick?.addEventListener('click', () => fileInput?.click());
    fileInput?.addEventListener('change', () => {
      fileObj = fileInput.files?.[0] || null;
      if (btnRun) btnRun.disabled = !fileObj;
      showError('');
      results?.classList.add('hidden');
      emptyBox?.classList.add('hidden');
      hideLowConfidenceUi();
      if (fileObj) {
        const r = new FileReader();
        r.onload = () => {
          if (preview) {
            preview.src = r.result;
            preview.classList.remove('hidden');
          }
          previewEmpty?.classList.add('hidden');
          uploadZone?.classList.add('has-image');
        };
        r.readAsDataURL(fileObj);
      }
    });

    btnRun?.addEventListener('click', async () => {
      if (!fileObj) return;
      showError('');
      results?.classList.add('hidden');
      emptyBox?.classList.add('hidden');
      hideLowConfidenceUi();
      setSpinner(true);
      const fd = new FormData();
      fd.append('file', fileObj, fileObj.name);
      try {
        const res = await fetch('/api/projects/' + pid + '/search', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error_code === 'project_empty' || data.error_code === 'no_index') {
          emptyBox?.classList.remove('hidden');
          const msg = $('emptyProjectMsg');
          if (msg) msg.textContent = data.message;
          const link = $('emptyProjectLink');
          if (link && data.action_url) {
            link.href = data.action_url;
            link.textContent = data.error_code === 'no_index' ? 'Открыть галерею' : 'Добавить особь';
          }
          return;
        }
        if (!data.success) {
          showError(data.error || 'Ошибка');
          return;
        }
        const unwrapped = $('unwrapped');
        if (data.unwrapped_base64 && unwrapped) {
          unwrapped.src = 'data:' + (data.unwrapped_mime || 'image/jpeg') + ';base64,' + data.unwrapped_base64;
        }
        const parts = [];
        if (data.best_match != null) parts.push('Лучшее совпадение: ' + data.best_match);
        if (data.confidence != null) parts.push('Уверенность: ' + data.confidence);
        if (data.is_new) parts.push('Ниже порога идентификации');
        const meta = $('meta');
        if (meta) meta.textContent = parts.join(' · ') || '—';

        lowConfDismissed = false;
        updateLowConfidenceUi(data.confidence);

        const tbody = $('tbody');
        if (tbody) {
          tbody.replaceChildren();
          (data.top_20_candidates || []).forEach((row) => {
            const tr = document.createElement('tr');
            const src = row.gallery_preview_path;
            tr.innerHTML =
              '<td><img src="' + src + '" class="table-thumb" loading="lazy" alt=""/></td>' +
              '<td><a href="' + (row.card_url || '#') + '" class="table-id">' + row.id + '</a></td>' +
              '<td><span class="table-score">' + row.score + '</span></td>' +
              '<td><a href="' + (row.card_url || '#') + '" class="table-link">Открыть карточку</a></td>';
            tbody.appendChild(tr);
          });
        }
        results?.classList.remove('hidden');
      } catch (e) {
        showError(e.message || 'Сеть недоступна');
      } finally {
        setSpinner(false);
      }
    });
  }

  /* ---------- individual form (create) ---------- */
  if (window.PAGE === 'individual_form' && window.TEMPLATE_SCHEMA) {
    const container = $('metadataFields');
    renderMetadataFields(container, window.TEMPLATE_SCHEMA, window.INITIAL_METADATA || {}, {
      projectId: window.PROJECT_ID,
    });

    apiJson('/api/projects/' + window.PROJECT_ID + '/individuals').then((data) => {
      const ids = (data.individuals || []).map((i) => i.individual_id);
      renderMetadataFields(container, window.TEMPLATE_SCHEMA, window.INITIAL_METADATA || {}, {
        projectId: window.PROJECT_ID,
        allIds: ids,
      });
    });

    $('individualForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const meta = collectMetadata(container, window.TEMPLATE_SCHEMA);
      const fd = new FormData();
      fd.append('metadata', JSON.stringify(meta));
      const files = $('photoFiles')?.files;
      if (files) {
        for (let i = 0; i < files.length; i++) fd.append('photos', files[i]);
      }
      try {
        const res = await fetch('/api/projects/' + window.PROJECT_ID + '/individuals', {
          method: 'POST',
          body: fd,
        });
        const data = await res.json();
        if (!res.ok) {
          const d = data.detail;
          throw new Error(typeof d === 'string' ? d : JSON.stringify(d) || 'Ошибка');
        }
        window.location.href = '/projects/' + window.PROJECT_ID + '/individuals/' + data.individual_id;
      } catch (err) {
        showToast(err.message, true);
      }
    });
  }

  /* ---------- individual card view/edit ---------- */
  if (window.PAGE === 'individual' && window.TEMPLATE_SCHEMA) {
    const cardFields = $('cardFields');
    const editFields = $('editFields');
    const cardView = $('cardView');
    const cardEdit = $('cardEdit');

    function renderView() {
      cardFields.replaceChildren();
      (window.TEMPLATE_SCHEMA.fields || []).forEach((field) => {
        if (field.key === 'individual_id') return;
        const dt = document.createElement('dt');
        dt.textContent = field.label_ru;
        const dd = document.createElement('dd');
        const val = window.METADATA[field.key] || '';
        if (field.type === 'individual_ref' && val) {
          const a = document.createElement('a');
          a.href = '/projects/' + window.PROJECT_ID + '/individuals/' + encodeURIComponent(val);
          a.className = 'field-ref-link';
          a.textContent = val;
          dd.appendChild(a);
        } else {
          dd.textContent = val || '—';
        }
        cardFields.appendChild(dt);
        cardFields.appendChild(dd);
      });
    }
    renderView();

    $('btnEdit')?.addEventListener('click', () => {
      cardView?.classList.add('hidden');
      cardEdit?.classList.remove('hidden');
      renderMetadataFields(editFields, window.TEMPLATE_SCHEMA, window.METADATA, {
        allIds: (window.ALL_IDS || []).filter((id) => id !== window.INDIVIDUAL_ID),
      });
    });

    $('btnCancelEdit')?.addEventListener('click', () => {
      cardEdit?.classList.add('hidden');
      cardView?.classList.remove('hidden');
    });

    $('editForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const meta = collectMetadata(editFields, window.TEMPLATE_SCHEMA);
      meta.individual_id = window.INDIVIDUAL_ID;
      try {
        const updated = await apiJson(
          '/api/projects/' + window.PROJECT_ID + '/individuals/' + encodeURIComponent(window.INDIVIDUAL_ID),
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ metadata: meta }),
          }
        );
        window.METADATA = updated.metadata;
        renderView();
        cardEdit?.classList.add('hidden');
        cardView?.classList.remove('hidden');
        showToast('Сохранено');
      } catch (err) {
        showToast(err.message, true);
      }
    });

    $('addPhotosForm')?.addEventListener('submit', async (e) => {
      e.preventDefault();
      const files = $('morePhotos')?.files;
      if (!files?.length) return;
      const fd = new FormData();
      for (let i = 0; i < files.length; i++) fd.append('photos', files[i]);
      try {
        await apiJson(
          '/api/projects/' + window.PROJECT_ID + '/individuals/' + encodeURIComponent(window.INDIVIDUAL_ID) + '/photos',
          { method: 'POST', body: fd }
        );
        showToast('Фото добавлены');
        location.reload();
      } catch (err) {
        showToast(err.message, true);
      }
    });

    $('btnDelete')?.addEventListener('click', async () => {
      if (!confirm('Удалить особь «' + window.INDIVIDUAL_ID + '»?')) return;
      try {
        const res = await fetch(
          '/api/projects/' + window.PROJECT_ID + '/individuals/' + encodeURIComponent(window.INDIVIDUAL_ID),
          { method: 'DELETE' }
        );
        if (!res.ok) throw new Error('Не удалось удалить');
        window.location.href = '/projects/' + window.PROJECT_ID + '/gallery';
      } catch (err) {
        showToast(err.message, true);
      }
    });
  }
})();
