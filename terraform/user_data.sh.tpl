#!/bin/bash
# Se ejecuta una sola vez al arrancar la instancia (cloud-init).
# Log completo en /var/log/cloud-init-output.log si algo falla.
set -euxo pipefail

apt-get update -y
apt-get install -y git ca-certificates curl

# docker-compose-plugin no está en los repos estándar de Ubuntu 22.04; el
# script oficial de Docker instala docker-ce + el plugin de compose (v2) bien.
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

cd /home/ubuntu
git clone ${repo_url} app
cd app

cat > .env <<EOF
ADMIN_TOKEN=${admin_token}
EOF

# compose.yml publica la API solo en 127.0.0.1 (pensado para uso local); este
# override la expone en todas las interfaces para que sea accesible desde
# fuera de la instancia.
cat > docker-compose.ec2.yml <<'EOF'
${ec2_override}
EOF

# Solo api + cron: notebook/pycaret necesita mucha más RAM de la que trae
# una t3.micro. Si subes el instance_type puedes añadir "mlflow notebook".
docker compose -f compose.yml -f docker-compose.ec2.yml up -d --build api cron

chown -R ubuntu:ubuntu /home/ubuntu/app
