/**
 * EpidermAI — Login / registro
 */
function setupAuthForm({ formId, endpoint, errorId, submitId, buildPayload, validate }) {
    const form = document.getElementById(formId);
    const errorBox = document.getElementById(errorId);
    const submitBtn = document.getElementById(submitId);
    const originalHtml = submitBtn.innerHTML;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorBox.classList.remove('visible');
        errorBox.textContent = '';

        if (validate) {
            const validationError = validate();
            if (validationError) {
                errorBox.textContent = validationError;
                errorBox.classList.add('visible');
                return;
            }
        }

        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-small"></span> Procesando...';

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(buildPayload()),
            });
            const data = await res.json();

            if (!res.ok) {
                errorBox.textContent = data.error || 'Ha ocurrido un error inesperado.';
                errorBox.classList.add('visible');
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalHtml;
                return;
            }

            window.location.href = '/';
        } catch (err) {
            errorBox.textContent = 'No se pudo conectar con el servidor.';
            errorBox.classList.add('visible');
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalHtml;
        }
    });
}

function setupPasswordToggle(buttonId, inputId) {
    const btn = document.getElementById(buttonId);
    const input = document.getElementById(inputId);
    if (!btn || !input) return;
    btn.addEventListener('click', () => {
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        btn.innerHTML = isPassword ? '<i data-lucide="eye-off"></i>' : '<i data-lucide="eye"></i>';
        if (window.lucide) lucide.createIcons();
    });
}
