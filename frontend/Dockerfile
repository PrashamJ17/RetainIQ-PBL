FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy application source code
COPY . .

# Build output
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
