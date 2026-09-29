(() => {
  'use strict';
  const form=document.querySelector('#sign-in'), status=document.querySelector('#login-status');
  form.addEventListener('submit',async event=>{
    event.preventDefault(); const button=form.querySelector('button');button.disabled=true;status.textContent='Signing in…';
    try {
      const initial=await fetch('/operator-ui/login',{credentials:'same-origin',cache:'no-store'});
      if(!initial.ok)throw new Error('Login unavailable.');
      const token=(await initial.json()).csrf_token;
      if(typeof token!=='string')throw new Error('Login unavailable.');
      const body=new URLSearchParams(new FormData(form));body.set('csrf_token',token);
      const response=await fetch('/operator-ui/login',{method:'POST',credentials:'same-origin',cache:'no-store',body});
      const value=await response.json();
      if(response.ok&&value.classification==='NON_OPERATIONAL/AUTHENTICATED')location.assign('/operator-ui/forecasts');
      else status.textContent=response.status===401?'Username or password was not accepted.':response.status===400?'Secure session unavailable. Open this page through the existing private connection or localhost, then try again.':'Login unavailable. Please try again later.';
    } catch(error){status.textContent='Cannot reach the login service.';}
    finally {button.disabled=false;}
  });
})();
