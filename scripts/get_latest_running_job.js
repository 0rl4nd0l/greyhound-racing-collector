const fs = require('fs');
let data = '';
process.stdin.on('data', chunk => data += chunk).on('end', () => {
  try {
    const j = JSON.parse(data || '{}');
    const jobs = j.training_jobs || {};
    const arr = Object.values(jobs).filter(x => x && x.status === 'running');
    arr.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    process.stdout.write(arr[0] ? (arr[0].id || '') : '');
  } catch (e) {
    process.stdout.write('');
  }
});

